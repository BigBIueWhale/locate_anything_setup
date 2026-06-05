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
    pts, off = parsing.parse_points("<box><10><20></box>", W, H)
    assert pts == [] and off == 1


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

def _route(answer, expected_shape):
    """Mirror inference.py's A.3 arithmetic over the parser primitives."""
    boxes, box_off = parsing.parse_boxes(answer, W, H)
    points, point_off = parsing.parse_points(answer, W, H)
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
