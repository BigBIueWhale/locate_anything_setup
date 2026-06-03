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

The literal `None` (capital N, token id 4064 in the released checkpoint)
inside a box is the per-category abstention marker. Note that NVIDIA's
DATA_PREPARATION.md:143 shows lowercase `none` but the trained token
decodes to capital `None`; our regexes are numeric-only so `<box>None</box>`
is silently dropped regardless of case — identical to NVIDIA's eval parser
at `Embodied/evaluation/inference_grounding_ddp.py:282-300` (in the
NVlabs/Eagle repo) which uses the same numeric-only `box_pattern`. The
lowercase variant is also tolerated by `has_abstention` as a forward-
compat safety net.

Multi-instance grounding (template 3) emits ONE <ref> followed by N sibling
<box> blocks; all N boxes are instances of the same phrase. Our parser mirrors
NVIDIA's eval-time `<ref>(category)</ref>((?:<box>.*?</box>)+)` capture
(`Embodied/evaluation/inference_grounding_ddp.py:390` and
`inference_detection_ddp.py:282-300`) so every sibling box inherits the
ref's label.

Aggregate abstention ("the frame returned nothing usable") means the model
produced NO parseable geometry at all — no boxes and no points. The response's
`abstained` field at `InferenceResult.abstained` is derived in
`worker/inference.py` from the PRE-(task-shape-)filter parse, so an off-shape
emission (geometry in the wrong shape for the task) is reported as a deviation
(`off_shape_count`), NOT as abstention. It deliberately does NOT scan raw_text
for `<box>None</box>`, because per-category abstention triples are emitted
alongside real detections in multi-category prompts and a substring scan would
flip the aggregate flag to True even when other categories returned valid
boxes. NVIDIA's own pipeline has no aggregate `abstained` concept either —
`metrics/other_metric.py:140-156` only tracks per-category None.

`has_abstention` is retained as a substring utility used by the BOOT SELF-TEST
in `worker/calibration.py` to distinguish "model emitted the trained explicit
abstention literal" from "model emitted gibberish the parser couldn't consume".
It is NOT the truth source for the response field.

Verified against NVlabs/Eagle's `Embodied/locateanything_worker.py`
(LocateAnythingWorker.parse_boxes) and the model's `generate_utils.py`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
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
# Used both standalone (orphan-box pass-2 fallback) and inside the captured
# boxes group of _REF_RUN_RE.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# Regex for a single <box>...</box> with exactly 2 numeric coords (point).
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
# Regex for explicit None abstention. The model emits capital-N `None` (mirroring
# the Python literal); we also accept lowercase as a forward-compat safety net.
_NONE_RE = re.compile(r"<box>[Nn]one</box>")


@dataclass(frozen=True)
class Detection:
    """A bounding-box detection with its label (if any)."""
    label: Optional[str]
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
    label: Optional[str]
    point_norm: list  # [x, y]
    point_px:   list  # [x, y] in pixels relative to source image

    def to_json(self) -> dict:
        return {"label": self.label, "point_norm": self.point_norm, "point_px": self.point_px}


def parse_boxes(answer: str, image_width: int, image_height: int) -> List[Detection]:
    """
    Parse all <box>...</box> blocks (with optional preceding <ref>) into
    Detection objects. Coordinates are scaled to the SOURCE image size.

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
          extract every valid 4-coord <box> inside via _BOX_RE, attributing
          all of them to the run's label. This is the shape NVIDIA trained
          on for templates 1-6 (one <ref> followed by ≥1 siblings; template 3
          is the only one that legitimately emits multiple siblings).
      (2) Find orphan <box> blocks — boxes whose span does NOT overlap any
          ref-run from pass 1. Attribute them as label=None. None of the
          seven canonical templates emit bare boxes, but we accept them
          defensively for off-pattern model output.

    Interval-overlap check: `a.start < b.end AND b.start < a.end`.
    """
    out: List[Detection] = []
    consumed_intervals: List[tuple] = []

    # Pass 1: each <ref>…</ref><box>…</box>[<box>…</box>…] run emits one
    # Detection per VALID box inside the run, ALL sharing the ref's label.
    for m in _REF_RUN_RE.finditer(answer):
        consumed_intervals.append(m.span())
        label = m.group("label").strip() or None
        for bm in _BOX_RE.finditer(m.group("boxes")):
            x1, y1, x2, y2 = (int(bm.group(k)) for k in (1, 2, 3, 4))
            # No geometric reject: `_make_box` clamps to [0,1000] and
            # canonicalizes corner order (min/max), so a box the model
            # localized is never silently dropped (matches NVIDIA's eval).
            out.append(_make_box(label, x1, y1, x2, y2,
                                 image_width, image_height))

    # Pass 2: orphan boxes — boxes whose span does NOT overlap any ref-run.
    for m in _BOX_RE.finditer(answer):
        s, e = m.span()
        if any(s < ie and is_ < e for (is_, ie) in consumed_intervals):
            continue
        x1, y1, x2, y2 = (int(m.group(k)) for k in (1, 2, 3, 4))
        out.append(_make_box(None, x1, y1, x2, y2, image_width, image_height))

    return out


def parse_points(answer: str, image_width: int, image_height: int) -> List[Point]:
    """Parse <box><x><y></box> point blocks. Two coords only.

    Mirrors NVIDIA's eval-time parser at
    NVlabs/Eagle's `Embodied/evaluation/inference_grounding_ddp.py:564-587`,
    which runs BOTH a point_pattern AND a box_pattern over each captured
    ref-run, attaching the run's category to every match.

    Two-pass design:
      (1) For each <ref>label</ref><box>...</box>... ref-run, extract every
          valid 2-coord <box> inside via _POINT_RE, attributing them to the
          run's label. Template 7 (`Point to: PHRASE.`) emits this shape:
          `<ref>PHRASE</ref><box><x><y></box>` — single labeled point per
          query.
      (2) Orphan points — 2-coord blocks not inside any ref-run and not
          shadowed by a 4-coord box span. These get label=None. The model
          can also emit bare points without a <ref> prefix.

    De-dup rule: a 2-coord <box><x><y></box> is a strict substring of the
    `<box><x><y>...` prefix of any 4-coord box, but _POINT_RE requires
    `</box>` immediately after the 2nd coord, so it can never spuriously
    match inside a real 4-coord block's text. We still dedup against
    _BOX_RE spans defensively — and we dedup pass-2 against pass-1's
    point spans to avoid double-counting points that live inside ref-runs.

    Critically, we do NOT dedup against _REF_RUN_RE spans whole — the
    ref-run span contains the box content, and for template 7 the box
    content IS the point we want to extract.
    """
    box_spans: List[tuple] = [m.span() for m in _BOX_RE.finditer(answer)]
    out: List[Point] = []
    consumed_point_spans: List[tuple] = []

    # Pass 1: labeled points inside <ref>...</ref><box>...</box>... ref-runs.
    for m in _REF_RUN_RE.finditer(answer):
        label = m.group("label").strip() or None
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
            x, y = int(pm.group(1)), int(pm.group(2))
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                continue
            out.append(
                Point(
                    label=label,
                    point_norm=[x, y],
                    point_px=[round(x / 1000.0 * image_width, 2),
                              round(y / 1000.0 * image_height, 2)],
                )
            )
            consumed_point_spans.append((abs_start, abs_end))

    # Pass 2: orphan points — 2-coord blocks not inside any 4-coord box
    # and not already emitted by pass 1.
    for m in _POINT_RE.finditer(answer):
        s, e = m.span()
        if any(s < ie and bs < e for (bs, ie) in box_spans):
            continue
        if any(s < ie and cs < e for (cs, ie) in consumed_point_spans):
            continue
        x, y = int(m.group(1)), int(m.group(2))
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            continue
        out.append(
            Point(
                label=None,
                point_norm=[x, y],
                point_px=[round(x / 1000.0 * image_width, 2),
                          round(y / 1000.0 * image_height, 2)],
            )
        )
    return out


def has_abstention(answer: str) -> bool:
    """Return True if `answer` contains the trained explicit abstention
    literal `<box>None</box>` (or lowercase `<box>none</box>` as a
    forward-compat tolerance).

    This is a SUBSTRING TEST — it fires once per match regardless of how
    many other categories returned real boxes in the same response. Use
    it ONLY when you specifically want to know whether the model emitted
    the trained abstention literal at all (e.g. in `worker/calibration.py`
    to distinguish "model produced recognized output" from "model emitted
    gibberish" at boot). For the aggregate "did this frame return
    anything usable" question, use `InferenceResult.abstained` (derived in
    `worker/inference.py` from the pre-(task-shape-)filter parse) — it
    matches NVIDIA's eval pipeline which has no aggregate abstained concept
    (see module docstring)."""
    return _NONE_RE.search(answer) is not None


def _clamp_coord(c: int) -> int:
    """Clamp a coordinate to the valid [0, 1000] token grid. Defense-in-depth:
    the 1001 coord tokens <0>..<1000> already bound the regex captures, so this
    is a no-op on conforming output — but it guarantees the [0,1000] invariant
    holds even if a future model/tokenizer change widens the captured range,
    rather than emitting an out-of-grid box."""
    return 0 if c < 0 else 1000 if c > 1000 else c


def _make_box(
    label: Optional[str],
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
