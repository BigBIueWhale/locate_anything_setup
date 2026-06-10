"""
Parse LocateAnything-3B output text into structured detections.

The model emits 6-token blocks like:
    <box><x1><y1><x2><y2></box>            — box, 4 integer coords in [0,1000]
    <box><x><y></box>                       — point, 2 integer coords in [0,1000]
    <box>None</box>                         — per-category abstention placeholder
    <ref>category</ref><box>...</box>       — labeled box (templates 1, 2, 5, 6)
    <ref>phrase</ref><box>A</box><box>B</box><box>C</box>
                                            — multi-instance grounding
                                              (template 3): ONE <ref> followed by
                                              N sibling <box> blocks, ALL
                                              sharing the label

The literal `None` inside a box (`<box>None</box>`) is the per-category
abstention marker. It is NOT a dedicated special token — the released
checkpoint has no `<None>`/`None` added token; `None` decodes as ordinary
text sub-words between the `<box>`/`</box>` tags (verified against the model's
`added_tokens.json` at the pinned revision). NVIDIA's DATA_PREPARATION.md:143
shows lowercase `none`, but the trained output decodes to capital `None`;
`_NONE_RE` matches both as a forward-compat safety net. `<box>None</box>` is
an on-contract "this category is absent" signal — it is neither emitted as a
detection nor counted as a deviation (it matches none of the numeric
coordinate regexes), mirroring NVIDIA's eval parser at
`Embodied/evaluation/inference_grounding_ddp.py:282-300` (NVlabs/Eagle).

Multi-instance grounding (template 3) emits ONE <ref> followed by N sibling
<box> blocks; all N boxes are instances of the same phrase. Our parser mirrors
NVIDIA's eval-time `<ref>(category)</ref>((?:<box>.*?</box>)+)` capture
(`Embodied/evaluation/inference_grounding_ddp.py:390` and
`inference_detection_ddp.py:282-300`) so every sibling box inherits the
ref's label.

CONTRACT (wire v2): every emitted Detection/Point carries a non-empty label —
a label-less geometry is unrepresentable on the wire
(rust_server/src/protocol.rs::LabeledBox/LabeledPoint). The label's SOURCE is
shape-specific:
  * BOX templates (1-6) emit an explicit `<ref>label</ref>` before every box
    run, so a box is labeled by its `<ref>`.
  * The POINT template (7, `Point to: PHRASE.`) is the exception: its TRAINED
    output is a BARE `<box><x><y></box>` with NO `<ref>` (DATA_PREPARATION.md:
    195). `parse_points` labels each bare point with the `point_label` it is
    given — the queried phrase, recovered via `prompts.point_phrase` and passed
    in by `worker/inference.py` for the `point` task — mirroring NVIDIA's eval,
    which attributes a single-category pointing call's bare points to the one
    queried category (inference_grounding_ddp.py:297-312). A point that instead
    arrives inside a `<ref>` run is labeled by that ref.
Off-contract shapes appear only in NON-conforming output and are explicitly NOT
turned into label=None geometry:
  * an ORPHAN box, or a bare point on a BOX task — geometry with no `<ref>` and
    no queried-phrase label to attribute it to (a bare point on the `point`
    task is NOT orphan: it is the trained output, labeled as above);
  * an EMPTY-REF box/point — one inside a `<ref></ref>` run whose label
    strips to the empty string; and
  * a MALFORMED-ARITY block — a `<box>…</box>` whose contents are neither a
    4-coord box, a 2-coord point, nor `None` (e.g. a 3- or 5-coord arity slip
    that pure-AR / `slow` decoding can produce). NVIDIA's eval parser drops
    these SILENTLY; we instead COUNT them via `count_malformed_geometry` so a
    localization the model attempted can never vanish without a trace.
The orphan/empty-ref counts ride the `off_contract_count` second element of the
(geometry, off_contract_count) tuple `parse_boxes`/`parse_points` return; the
malformed-arity count is returned separately by `count_malformed_geometry`.
`worker/inference.py` applies the A.3 mapping over these: it keeps the valid
geometry, folds EVERY off-contract count (orphan/empty-ref + cross-shape
geometry for the task + malformed-arity) into `deviations_dropped`, abstains
only on zero geometry of any kind, and emits a `model_deviation` error
whenever geometry was present (valid, cross-shape, or malformed) but zero of
it was valid for the task — so a frame whose ONLY geometry is malformed is a
loud `model_deviation`, never a silent `abstained`. The verbatim token-order
emission is always recoverable from `InferenceResult.raw_answer`.

`has_abstention` is retained as a substring utility used by the BOOT SELF-TEST
in `worker/calibration.py` to distinguish "model emitted the trained explicit
abstention literal" from "model emitted gibberish the parser couldn't consume".

Verified against NVlabs/Eagle's `Embodied/locateanything_worker.py`
(LocateAnythingWorker.parse_boxes) and the model's `generate_utils.py`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

# Regex for a <ref>label</ref> ref-run followed by ONE OR MORE sibling <box>
# blocks. Mirrors NVIDIA's eval parser in the NVlabs/Eagle repo at
#   Embodied/evaluation/inference_grounding_ddp.py:390
#     ref_pattern = r'<ref>([^<]+)</ref>((?:<box>.*?</box>)+)'
#   Embodied/evaluation/inference_detection_ddp.py:282-300
#     (same pattern; their inline example shows two boxes attributed to one ref)
#
# The shared-ref-multi-box shape is the TRAINED output for template 3 (phrase
# grounding multi); see DATA_PREPARATION.md:171 verbatim:
#     <ref>people wearing hats</ref><box><100>...</box><box><500>...</box>
# Template 1 (closed-class detection) re-emits <ref> per box per
# DATA_PREPARATION.md:155 so each category-group has its own <ref>X</ref>
# followed by its own box-run — still correctly handled by this regex
# because the lazy box-run terminates at the next <ref>. Template 5
# (scene-text) similarly re-emits <ref>text</ref> per box per
# DATA_PREPARATION.md:179.
#
# Inner pattern <box>.*?</box> is lazy (non-greedy) with re.DOTALL so it
# matches the shortest `<box>…</box>` span possible — important when the model
# emits per-category abstention `<box>None</box>` interspersed with real
# boxes. _BOX_RE (the numeric-only 4-coord regex) is used to extract real
# boxes from the captured run; `None` blocks are silently dropped because
# they don't match _BOX_RE.
_REF_RUN_RE = re.compile(
    r"<ref>(?P<label>[^<]*?)</ref>"
    r"(?P<boxes>(?:\s*<box>.*?</box>)+)",
    flags=re.DOTALL,
)
# Regex for a single <box>...</box> with exactly 4 numeric coords.
# Used both standalone (orphan-box pass-2 detection) and inside the captured
# boxes group of _REF_RUN_RE.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# Regex for a single <box>...</box> with exactly 2 numeric coords (point).
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
# Regex for explicit None abstention. The model emits capital-N `None` (mirroring
# the Python literal); we also accept lowercase as a forward-compat safety net.
_NONE_RE = re.compile(r"<box>[Nn]one</box>")
# Regex for ANY <box>…</box> block, valid or not (lazy + DOTALL). Used by
# `count_malformed_geometry` to surface geometry-shaped blocks that match none
# of the on-contract numeric forms above (a malformed-arity localization).
# Requires a closing </box>, so a TRUNCATED final block (open `<box>` with no
# `</box>`) is NOT matched here — that case is surfaced by
# `model_output_truncated` instead, not double-counted as a deviation.
_ANY_BOX_RE = re.compile(r"<box>.*?</box>", flags=re.DOTALL)


@dataclass(frozen=True)
class Detection:
    """A bounding-box detection with its (required, non-empty) label.

    Per the wire-v2 contract a Detection is ONLY constructed for a box that
    carries a non-empty `<ref>` label; orphan / empty-ref boxes are
    off-contract and never become a Detection (see module docstring). The
    label is therefore non-optional."""
    label: str
    # Canonical [0, 1000] integer box: each coord clamped to the 1001-token
    # grid (<0>..<1000>) and corners min/max-sorted so x1<=x2 and y1<=y2. The
    # model decodes the 4 coordinate positions INDEPENDENTLY (no monotonicity
    # constraint — see `_make_box`), so raw emission order carries no meaning;
    # we canonicalize it exactly as NVIDIA's eval does and never drop a box on
    # corner order. The VERBATIM token-order emission is always recoverable
    # from InferenceResult.raw_answer.
    bbox_norm: list  # [x1, y1, x2, y2], canonical (x1<=x2, y1<=y2)
    bbox_px: list    # [x1, y1, x2, y2] in pixels relative to source image

    def to_json(self) -> dict:
        return {"label": self.label, "bbox_norm": self.bbox_norm, "bbox_px": self.bbox_px}


@dataclass(frozen=True)
class Point:
    """A point detection with its (required, non-empty) label.

    A Point's label is either its enclosing `<ref>` (when the model emits a
    point inside a ref-run) or — for the `point` task, whose trained output is
    a bare `<box><x><y></box>` with no `<ref>` — the queried phrase supplied as
    `parse_points(point_label=…)`. The label is never empty; a 2-coord block
    with neither label source (a bare point on a box task) is off-contract and
    never becomes a Point (see module docstring)."""
    label: str
    point_norm: list  # [x, y]
    point_px:   list  # [x, y] in pixels relative to source image

    def to_json(self) -> dict:
        return {"label": self.label, "point_norm": self.point_norm, "point_px": self.point_px}


def parse_boxes(
    answer: str, image_width: int, image_height: int
) -> Tuple[List[Detection], int]:
    """Parse `<ref>label</ref><box>...</box>` runs into Detection objects.

    Returns a `(detections, off_contract_count)` tuple:
      * `detections` — every VALID labeled box (non-empty `<ref>` label),
        coordinates scaled to the SOURCE image size and canonicalized by
        `_make_box` (clamp to [0,1000] + corner-sort). Degenerate / zero-area
        boxes are kept (NVIDIA forces valid-detection coords to 0 and keeps
        them); a box is NEVER dropped on corner order or range.
      * `off_contract_count` — the number of boxes that are off-contract for
        the wire-v2 shape and were therefore NOT emitted: orphan boxes (no
        preceding `<ref>` run) and empty-ref boxes (`<ref></ref>` label strips
        to empty). `worker/inference.py` folds this into `deviations_dropped`.

    Note: the LocateAnything README's parse_boxes uses the ORIGINAL image
    width/height. The model emits coords relative to whatever image it
    actually saw (post-resize). The processor's 28-px-grid step is an
    ANAMORPHIC resize (independent ceil-to-28 per axis, so the x and y scale
    factors differ slightly — it is NOT a uniform scale, and NOT a pad). The
    [0,1000]→[0,W] map is exact nonetheless because each axis is normalized
    INDEPENDENTLY: coord/1000 × source_w for x, coord/1000 × source_h for y.
    Passing src dims is therefore correct in either orientation.

    Two-pass design:
      (1) Find each <ref>label</ref> ref-run via _REF_RUN_RE. For each run,
          extract every valid 4-coord <box> inside via _BOX_RE. If the run's
          label is non-empty, attribute all of them to that label (valid
          detections); if the label strips to empty, the boxes are
          off-contract and only counted. This is the shape NVIDIA trained on
          for templates 1-6 (one <ref> followed by ≥1 siblings; template 3 is
          the only one that legitimately emits multiple siblings).
      (2) Find orphan <box> blocks — boxes whose span does NOT overlap any
          ref-run from pass 1. None of the seven canonical templates emit
          bare boxes; an orphan box is off-contract and only counted.

    Interval-overlap check: `a.start < b.end AND b.start < a.end`.
    """
    out: List[Detection] = []
    off_contract = 0
    consumed_intervals: List[tuple] = []

    # Pass 1: each <ref>…</ref><box>…</box>[<box>…</box>…] run. A non-empty
    # label yields one Detection per VALID box inside the run, all sharing the
    # label. An empty-ref run's boxes are off-contract (counted, not emitted).
    for m in _REF_RUN_RE.finditer(answer):
        consumed_intervals.append(m.span())
        label = m.group("label").strip()
        boxes_in_run = list(_BOX_RE.finditer(m.group("boxes")))
        if not label:
            # Empty `<ref></ref>` run: every valid box inside is off-contract
            # (no non-empty label to attribute it to).
            off_contract += len(boxes_in_run)
            continue
        for bm in boxes_in_run:
            x1, y1, x2, y2 = (int(bm.group(k)) for k in (1, 2, 3, 4))
            # No geometric reject: `_make_box` clamps to [0,1000] and
            # canonicalizes corner order (min/max), so a box the model
            # localized is never silently dropped (matches NVIDIA's eval).
            out.append(_make_box(label, x1, y1, x2, y2,
                                 image_width, image_height))

    # Pass 2: orphan boxes — boxes whose span does NOT overlap any ref-run.
    # Off-contract for wire v2 (no <ref> label): counted, never emitted.
    for m in _BOX_RE.finditer(answer):
        s, e = m.span()
        if any(s < ie and is_ < e for (is_, ie) in consumed_intervals):
            continue
        off_contract += 1

    return out, off_contract


def parse_points(
    answer: str, image_width: int, image_height: int,
    point_label: Optional[str] = None,
) -> Tuple[List[Point], int]:
    """Parse the model's pointing / grounding-point output into Point objects.

    Returns a `(points, off_contract_count)` tuple. Every emitted Point carries
    a non-empty label; where the label comes from depends on the shape the
    model emitted:

      * `<ref>label</ref><box><x><y></box>` (a point inside a ref-run) → the
        Point is labeled with the run's `<ref>` (the grounding-point shape;
        NVIDIA's grounding eval parses points from ref-runs at
        inference_grounding_ddp.py:564-587).
      * a BARE `<box><x><y></box>` (no `<ref>`) → this is the TRAINED output of
        template 7 `Point to: PHRASE.` (DATA_PREPARATION.md:195 maps it to a
        bare point with no ref). When `point_label` is supplied (the worker
        passes the queried phrase for the `point` task) each bare point is
        labeled with it — exactly how NVIDIA's eval attributes a single-
        category pointing call's bare points to the one queried category
        (inference_grounding_ddp.py:297-312). When `point_label` is None (the
        prompt was a BOX task) a bare point is cross-shape and off-contract:
        counted, never emitted.

    `point_label` is therefore the queried phrase for the `point` task and None
    for every other task; `worker/inference.py` supplies it via
    `prompts.point_phrase(prompt)`. The resulting non-empty label is what the
    wire contract requires (rust_server/src/protocol.rs::LabeledPoint.label).

    Two-pass design:
      (1) For each <ref>label</ref><box>...</box>... ref-run, extract every
          valid 2-coord <box> inside via _POINT_RE. A non-empty ref labels the
          Points; an empty-ref run's points are off-contract (counted).
      (2) Bare points — 2-coord blocks not inside any ref-run and not shadowed
          by a 4-coord box span. Labeled with `point_label` (point task) or
          counted off-contract (box task / `point_label` None).

    De-dup rule: a 2-coord <box><x><y></box> is a strict substring of the
    `<box><x><y>...` prefix of any 4-coord box, but _POINT_RE requires
    `</box>` immediately after the 2nd coord, so it can never spuriously
    match inside a real 4-coord block's text. We still dedup against
    _BOX_RE spans defensively — and we dedup pass-2 against pass-1's
    point spans to avoid double-counting points that live inside ref-runs.

    Critically, we do NOT dedup against _REF_RUN_RE spans whole — the
    ref-run span contains the box content, and for a ref-labeled point the box
    content IS the point we want to extract.
    """
    box_spans: List[tuple] = [m.span() for m in _BOX_RE.finditer(answer)]
    out: List[Point] = []
    off_contract = 0
    consumed_point_spans: List[tuple] = []

    # Pass 1: points inside <ref>...</ref><box>...</box>... ref-runs. Non-empty
    # label → Points; empty-ref → off-contract (counted).
    for m in _REF_RUN_RE.finditer(answer):
        label = m.group("label").strip()
        boxes_text = m.group("boxes")
        boxes_offset = m.start("boxes")
        for pm in _POINT_RE.finditer(boxes_text):
            abs_start = boxes_offset + pm.start()
            abs_end   = boxes_offset + pm.end()
            # Skip if this 2-coord match overlaps a real 4-coord box span.
            # (Cannot happen in practice — _POINT_RE requires </box> after
            # 2nd coord — but defensive against future regex relaxation.)
            if any(abs_start < ie and bs < abs_end for (bs, ie) in box_spans):
                continue
            consumed_point_spans.append((abs_start, abs_end))
            if not label:
                # Empty-ref point: off-contract (no non-empty label).
                off_contract += 1
                continue
            # _make_point clamps each coord to the [0,1000] grid and KEEPS it —
            # a point the model localized is never silently dropped on range
            # (symmetric with the box path's `_make_box`).
            out.append(_make_point(label, int(pm.group(1)), int(pm.group(2)),
                                   image_width, image_height))

    # Pass 2: bare points — 2-coord blocks not inside any 4-coord box and not
    # already emitted/counted by pass 1. For the POINT task these ARE the
    # trained output (the model emits `<box><x><y></box>` with no `<ref>`); we
    # label each with `point_label`, the queried phrase, matching NVIDIA's eval
    # (a single-category pointing call's bare points all carry that one queried
    # category). For every other task `point_label` is None → a bare point is
    # cross-shape and off-contract: COUNTED, never silently dropped.
    for m in _POINT_RE.finditer(answer):
        s, e = m.span()
        if any(s < ie and bs < e for (bs, ie) in box_spans):
            continue
        if any(s < ie and cs < e for (cs, ie) in consumed_point_spans):
            continue
        if point_label:
            out.append(_make_point(point_label, int(m.group(1)),
                                   int(m.group(2)), image_width, image_height))
        else:
            off_contract += 1
    return out, off_contract


def count_malformed_geometry(answer: str) -> int:
    """Count `<box>…</box>` blocks that are geometry-shaped but match NONE of
    the on-contract forms: a 4-coord box (`_BOX_RE`), a 2-coord point
    (`_POINT_RE`), or the `<box>None</box>` per-category abstention (`_NONE_RE`).

    These are localization attempts the model emitted in a MALFORMED shape
    (e.g. a 3- or 5-coord block from an arity slip in pure-AR / `slow`
    decoding). They are off-contract and MUST be accounted for: `inference.py`
    folds this count into `deviations_dropped` and treats it as "geometry
    present", so a frame whose ONLY geometry is malformed reports
    `model_deviation` (loud) and NEVER a silent `abstained`, and a malformed
    block co-emitted with valid geometry is reflected in `deviations_dropped`
    rather than vanishing.

    This is deliberately STRICTER than NVIDIA's eval parser (NVlabs/Eagle),
    whose numeric-only `box_pattern` drops such blocks with no trace — the one
    place we are intentionally higher-fidelity than the reference, in service
    of the project's no-silent-drop contract.

    A truncated final block (open `<box>` with no closing `</box>`) is NOT
    counted here — `_ANY_BOX_RE` requires a `</box>`, so truncation is left to
    `model_output_truncated` and not double-signalled as a deviation.
    """
    n = 0
    for m in _ANY_BOX_RE.finditer(answer):
        block = m.group(0)
        if (
            _BOX_RE.fullmatch(block)
            or _POINT_RE.fullmatch(block)
            or _NONE_RE.fullmatch(block)
        ):
            continue
        n += 1
    return n


def has_abstention(answer: str) -> bool:
    """Return True if `answer` contains the trained explicit abstention
    literal `<box>None</box>` (or lowercase `<box>none</box>` as a
    forward-compat tolerance).

    This is a SUBSTRING TEST — it fires once per match regardless of how
    many other categories returned real boxes in the same response. Use
    it ONLY when you specifically want to know whether the model emitted
    the trained abstention literal at all (e.g. in `worker/calibration.py`
    to distinguish "model produced recognized output" from "model emitted
    gibberish" at boot). The aggregate "did this frame return anything
    usable" question is answered by the variant `worker/inference.py`
    selects (the `abstained` variant ⇔ zero parsed geometry of any kind);
    this substring probe is only the parser-drift self-test signal."""
    return _NONE_RE.search(answer) is not None


def _clamp_coord(c: int) -> int:
    """Clamp a coordinate to the valid [0, 1000] token grid. Defense-in-depth:
    the 1001 coord tokens <0>..<1000> already bound the regex captures, so this
    is a no-op on conforming output — but it guarantees the [0,1000] invariant
    holds even if a future model/tokenizer change widens the captured range,
    rather than emitting an out-of-grid box."""
    return 0 if c < 0 else 1000 if c > 1000 else c


def _make_box(
    label: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> Detection:
    """Build a Detection from four raw coordinate-token values.

    The model's decoder selects the four coordinate positions INDEPENDENTLY
    (per-position top-1, no monotonicity constraint — verified in the model's
    `generate_utils.py` `decode_bbox_avg`), so it can legitimately emit
    non-monotone corners (x1>x2 and/or y1>y2). NVIDIA's own evaluation
    (NVlabs/Eagle `Embodied/evaluation/inference_grounding_ddp.py:447-470`,
    `convert_normalized_bbox_to_absolute`) CLAMPS each coord and then min/max-
    sorts the corners, and KEEPS the box. We do exactly the same: a box the
    model localized must never be silently dropped on corner order or range.
    `bbox_norm`/`bbox_px` are therefore the canonical (clamped, corner-sorted)
    rectangle; the verbatim token-order emission stays in `raw_answer`."""
    x_lo, x_hi = sorted((_clamp_coord(x1), _clamp_coord(x2)))
    y_lo, y_hi = sorted((_clamp_coord(y1), _clamp_coord(y2)))
    return Detection(
        label=label,
        bbox_norm=[x_lo, y_lo, x_hi, y_hi],
        bbox_px=[
            round(x_lo / 1000.0 * image_width, 2),
            round(y_lo / 1000.0 * image_height, 2),
            round(x_hi / 1000.0 * image_width, 2),
            round(y_hi / 1000.0 * image_height, 2),
        ],
    )


def _make_point(
    label: str,
    x: int,
    y: int,
    image_width: int,
    image_height: int,
) -> Point:
    """Build a Point from two raw coordinate-token values.

    Mirrors `_make_box`: each coord is clamped to the [0,1000] token grid and
    KEPT — a point the model localized is never silently dropped on range. The
    model decodes the two coordinate positions independently (per the model's
    `generate_utils.py`), so the same clamp-and-keep discipline as the box path
    applies; the verbatim token-order emission stays in `raw_answer`."""
    xc, yc = _clamp_coord(x), _clamp_coord(y)
    return Point(
        label=label,
        point_norm=[xc, yc],
        point_px=[
            round(xc / 1000.0 * image_width, 2),
            round(yc / 1000.0 * image_height, 2),
        ],
    )
