"""
Unit tests for worker/parsing.py — the LocateAnything-3B detection-path parser.

Runs standalone (parsing.py imports only the standard library):
    python3 worker/test_parsing.py        # plain runner, no pytest needed
    pytest worker/test_parsing.py         # also works under pytest

These pin the no-silent-drop contract — in particular the malformed-arity
accounting that makes a localization the model attempted impossible to lose
silently (the fix is deliberately stricter than NVIDIA's eval parser, which
drops malformed blocks with no trace).
"""
from __future__ import annotations
import os
import sys

# Import worker/parsing.py directly (stdlib-only) without triggering the
# heavyweight worker package __init__ (torch/PIL).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parsing  # noqa: E402
import prompts  # noqa: E402

W, H = 1920, 1080


# ---- count_malformed_geometry: the headline fix -------------------------------

def test_lone_malformed_3coord_counted():
    assert parsing.count_malformed_geometry("<ref>cat</ref><box><10><20><30></box>") == 1


def test_lone_malformed_5coord_counted():
    assert parsing.count_malformed_geometry("<ref>cat</ref><box><1><2><3><4><5></box>") == 1


def test_empty_box_block_counted():
    assert parsing.count_malformed_geometry("<box></box>") == 1


def test_valid_box_not_counted():
    assert parsing.count_malformed_geometry("<ref>cat</ref><box><1><2><3><4></box>") == 0


def test_valid_point_not_counted():
    assert parsing.count_malformed_geometry("<ref>cat</ref><box><5><6></box>") == 0


def test_none_abstention_not_counted():
    assert parsing.count_malformed_geometry("<ref>book</ref><box>None</box>") == 0
    assert parsing.count_malformed_geometry("<box>none</box>") == 0


def test_truncated_open_block_not_counted():
    # No closing </box> → truncation, surfaced by model_output_truncated, not here.
    assert parsing.count_malformed_geometry("<ref>cat</ref><box><10><20") == 0


def test_mixed_counts_two_malformed():
    s = (
        "<ref>a</ref><box><1><2><3><4></box>"   # valid box      -> not counted
        "<ref>b</ref><box>None</box>"            # absent         -> not counted
        "<ref>c</ref><box><7><8><9></box>"       # malformed 3    -> counted
        "<box><1><2><3><4><5></box>"             # malformed 5    -> counted
    )
    assert parsing.count_malformed_geometry(s) == 2


# ---- parse_points: clamp-and-keep, never silent-drop --------------------------

def test_parse_points_clamps_out_of_range_instead_of_dropping():
    # A hypothetical >1000 coordinate is CLAMPED and KEPT (the old code silently
    # dropped it). Real tokens never exceed 1000; this pins defensive parity
    # with the box path (which clamps-and-keeps, never drops on range).
    pts, off = parsing.parse_points("<ref>drone</ref><box><1500><20></box>", W, H)
    assert off == 0
    assert len(pts) == 1
    assert pts[0].point_norm == [1000, 20]


def test_parse_points_valid():
    pts, off = parsing.parse_points("<ref>drone</ref><box><500><300></box>", W, H)
    assert len(pts) == 1 and off == 0
    assert pts[0].label == "drone" and pts[0].point_norm == [500, 300]


def test_orphan_point_counted_not_dropped():
    # Bare point with no point_label (a BOX task): cross-shape, off-contract —
    # counted, never silently dropped. On the `point` task the same bare point
    # is instead LABELED with the queried phrase; see the point-task tests below.
    pts, off = parsing.parse_points("<box><10><20></box>", W, H)
    assert pts == [] and off == 1


# ---- parse_points point-task labeling: the bare-point fix ---------------------

def test_bare_point_labeled_with_queried_phrase():
    # Template 7's TRAINED output is a bare <box><x><y></box> with no <ref>
    # (DATA_PREPARATION.md:195). On the point task it must be LABELED with the
    # queried phrase, not dropped — the bug this fix closes.
    pts, off = parsing.parse_points(
        "<box><950><30></box>", W, H, point_label="drone in the sky"
    )
    assert off == 0 and len(pts) == 1
    assert pts[0].label == "drone in the sky"
    assert pts[0].point_norm == [950, 30]


def test_multiple_bare_points_all_get_phrase():
    # Per-category pointing can return several points for the one queried
    # category; every bare point inherits that single phrase label.
    s = "<box><950><30></box><box><500><80></box><box><120><640></box>"
    pts, off = parsing.parse_points(s, W, H, point_label="drone")
    assert off == 0 and len(pts) == 3
    assert all(p.label == "drone" for p in pts)


def test_bare_point_clamped_and_labeled():
    # Defensive: an out-of-grid coord is clamped-and-kept, still labeled.
    pts, off = parsing.parse_points("<box><1500><30></box>", W, H, point_label="x")
    assert off == 0 and len(pts) == 1 and pts[0].point_norm == [1000, 30]


def test_ref_labeled_point_keeps_ref_not_point_label():
    # A point inside a <ref> run is labeled by that ref; point_label only labels
    # BARE points (pass 2), so a ref-labeled point is unaffected by it.
    pts, off = parsing.parse_points(
        "<ref>icon</ref><box><10><20></box>", W, H, point_label="ignored"
    )
    assert off == 0 and len(pts) == 1 and pts[0].label == "icon"


# ---- parse_boxes regression (unchanged behaviour) -----------------------------

def test_parse_boxes_valid_and_canonicalized():
    boxes, off = parsing.parse_boxes("<ref>cat</ref><box><560><640><420><510></box>", W, H)
    assert len(boxes) == 1 and off == 0
    # corners min/max-sorted so x1<=x2, y1<=y2
    assert boxes[0].bbox_norm == [420, 510, 560, 640]


def test_parse_boxes_none_is_absence():
    boxes, off = parsing.parse_boxes("<ref>book</ref><box>None</box>", W, H)
    assert boxes == [] and off == 0  # absent category: not a box, not a deviation


def test_parse_boxes_multi_category_some_absent():
    s = (
        "<ref>bottle</ref><box><117><233><235><758></box>"
        "<ref>book</ref><box>None</box>"
        "<ref>cup</ref><box><742><456><900><705></box>"
    )
    boxes, off = parsing.parse_boxes(s, W, H)
    assert len(boxes) == 2 and off == 0
    assert parsing.count_malformed_geometry(s) == 0


# ---- routing simulation: malformed-only -> model_deviation, not abstained -----

def _route(answer, expected_shape, point_label=None):
    """Mirror inference.py's A.3 arithmetic over the parser primitives.

    `point_label` is threaded into parse_points only for the point task, exactly
    as inference.py passes `prompts.point_phrase(prompt)` for the `point` task
    and None otherwise."""
    boxes, box_off = parsing.parse_boxes(answer, W, H)
    points, point_off = parsing.parse_points(
        answer, W, H, point_label=(point_label if expected_shape == "point" else None)
    )
    malformed = parsing.count_malformed_geometry(answer)
    if expected_shape == "point":
        valid = len(points)
        dropped = len(boxes) + box_off + point_off + malformed
    else:
        valid = len(boxes)
        dropped = len(points) + point_off + box_off + malformed
    any_geom = bool(boxes or points or box_off or point_off or malformed)
    if valid > 0:
        return ("success", dropped)
    return ("model_deviation" if any_geom else "abstained", dropped)


def test_lone_malformed_box_routes_to_model_deviation():
    # THE fix: a box-task frame whose only geometry is a malformed box used to
    # report `abstained, dropped=0` (silent-wrong). It must now be a loud
    # model_deviation that counts the dropped block.
    kind, dropped = _route("<ref>cat</ref><box><10><20><30></box>", "box")
    assert kind == "model_deviation" and dropped == 1


def test_pure_abstention_still_abstains():
    kind, dropped = _route("<ref>book</ref><box>None</box>", "box")
    assert kind == "abstained" and dropped == 0


def test_valid_plus_malformed_keeps_valid_and_counts():
    kind, dropped = _route(
        "<ref>cat</ref><box><10><20><30><40></box><box><1><2><3></box>", "box"
    )
    assert kind == "success" and dropped == 1


def test_point_task_bare_points_route_to_success():
    # THE fix end-to-end: the point task's trained bare-point output used to
    # route to model_deviation (0 points). With the queried phrase as the label
    # it must now be a success carrying the labeled point, dropping nothing.
    kind, dropped = _route("<box><950><30></box>", "point", point_label="drone")
    assert kind == "success" and dropped == 0


def test_point_task_cross_shape_box_still_deviates():
    # A 4-coord box under the point task is genuinely cross-shape (the model
    # gave a box when asked to point) → loud model_deviation, count the drop.
    kind, dropped = _route(
        "<ref>x</ref><box><10><20><30><40></box>", "point", point_label="drone"
    )
    assert kind == "model_deviation" and dropped == 1


def test_point_task_none_still_abstains():
    # `<box>None</box>` on the point task is the trained "nothing here" → abstain.
    kind, dropped = _route("<box>None</box>", "point", point_label="drone")
    assert kind == "abstained" and dropped == 0


# ---- prompts.point_phrase: byte-exact inverse of the point template -----------

def test_point_phrase_round_trips():
    assert prompts.point_phrase(prompts.point_to("drone in the sky")) == "drone in the sky"
    assert prompts.point_phrase("Point to: quadcopter.") == "quadcopter"


def test_point_phrase_preserves_interior_period():
    # The slot validator forbids a TRAILING '.' but allows interior ones; only
    # the template's single trailing period is removed.
    assert prompts.point_phrase("Point to: the U.S. flag.") == "the U.S. flag"


def test_point_phrase_rejects_non_point_prompt():
    try:
        prompts.point_phrase("Detect all the text in box format.")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on a non-point prompt")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e!r}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERR  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
