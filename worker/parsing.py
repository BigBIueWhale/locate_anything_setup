"""
Parse LocateAnything-3B output text into structured detections.

The model emits 6-token blocks like:
    <box><x1><y1><x2><y2></box>            — box, 4 integer coords in [0,1000]
    <box><x><y></box>                       — point, 2 integer coords in [0,1000]
    <box>None</box>                         — per-category abstention placeholder
    <ref>category</ref><box>...</box>       — labeled box (detection / grounding)

The literal `None` (capital N, token id 4064 in the released checkpoint —
verified live; NVIDIA's DATA_PREPARATION.md:143 shows lowercase `none` but the
trained token is capital `None`) inside a box is the per-category abstention
marker. Our box/point regexes are numeric-only, so `<box>None</box>` does not
match and is silently dropped — identical to NVIDIA's eval parser at
/tmp/nvlabs_eagle/Embodied/evaluation/inference_grounding_ddp.py:282-300 which
uses the same numeric-only box_pattern. The lowercase variant is also tolerated
by `has_abstention` as a forward-compat safety net.

Aggregate abstention ("the frame returned nothing usable") is derivable from
the parse results — empty `detections` AND empty `points` ⇔ aggregate
abstention. The response's `abstained` field at `InferenceResult.abstained` is
populated using exactly that derivation in `worker/inference.py`; it
deliberately does NOT scan raw_text for `<box>None</box>`, because per-category
abstention triples are emitted alongside real detections in multi-category
prompts and a substring scan would flip the aggregate flag to True even when
other categories returned valid boxes. NVIDIA's own pipeline has no aggregate
`abstained` concept either — `metrics/other_metric.py:140-156` only tracks
per-category None.

`has_abstention` is retained as a substring utility used by the BOOT SELF-TEST
in `worker/calibration.py` to distinguish "model emitted the trained explicit
abstention literal" from "model emitted gibberish the parser couldn't consume".
It is NOT the truth source for the response field.

Verified against /tmp/nvlabs_eagle/Embodied/locateanything_worker.py
(LocateAnythingWorker.parse_boxes) and the model's generate_utils.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import re

# Regex for a labeled box (ref-tag preceding the box block).
_LABELED_BOX_RE = re.compile(
    r"<ref>(?P<label>[^<]*?)</ref>"
    r"\s*<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)
# Regex for an unlabeled box.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# Regex for a point.
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
# Regex for explicit None abstention. The model emits capital-N `None` (mirroring
# the Python literal); we also accept lowercase as a forward-compat safety net.
_NONE_RE = re.compile(r"<box>[Nn]one</box>")


@dataclass(frozen=True)
class Detection:
    """A bounding-box detection with its label (if any)."""
    label: Optional[str]
    # Normalized [0, 1000] integer coords as emitted by the model. The model
    # quantizes spatial position into 1001 tokens — this is the canonical
    # representation. Pixel coords are derived per image.
    bbox_norm: list  # [x1, y1, x2, y2]
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
    actually saw (post-resize). Because the model's resize preserves
    aspect ratio and is uniform in x and y, the same [0,1000]→[0,W] map
    works for either dst or src, so passing src dims is correct.
    """
    out: List[Detection] = []
    consumed_intervals: List[tuple] = []

    # Pass 1: labeled boxes. Track each match's (start, end) so pass 2
    # can skip any unlabeled-box match whose <box>…</box> span falls
    # inside a labeled one (the labeled regex always consumes a strict
    # superset starting at <ref>).
    for m in _LABELED_BOX_RE.finditer(answer):
        consumed_intervals.append(m.span())
        x1, y1, x2, y2 = (int(m.group(k)) for k in ("x1", "y1", "x2", "y2"))
        if not _coord_valid(x1, y1, x2, y2):
            continue
        out.append(_make_box(m.group("label").strip() or None,
                             x1, y1, x2, y2, image_width, image_height))

    # Pass 2: unlabeled boxes — only those whose span does NOT overlap
    # any pass-1 span. Interval overlap is `a.start < b.end and b.start < a.end`.
    for m in _BOX_RE.finditer(answer):
        s, e = m.span()
        if any(s < ie and is_ < e for (is_, ie) in consumed_intervals):
            continue
        x1, y1, x2, y2 = (int(m.group(k)) for k in (1, 2, 3, 4))
        if not _coord_valid(x1, y1, x2, y2):
            continue
        out.append(_make_box(None, x1, y1, x2, y2, image_width, image_height))

    return out


def parse_points(answer: str, image_width: int, image_height: int) -> List[Point]:
    """Parse <box><x><y></box> point blocks. Two coords only.

    The point regex `<box><(\\d+)><(\\d+)></box>` would otherwise match
    spurious "point" blocks inside any 4-coord box (`<box><x1><y1>` is
    a prefix of `<box><x1><y1><x2><y2>`). We therefore filter out any
    point whose span overlaps a labeled or unlabeled box span. This
    mirrors the dedup pattern in parse_boxes.
    """
    consumed_intervals: List[tuple] = [
        m.span() for m in _LABELED_BOX_RE.finditer(answer)
    ] + [
        m.span() for m in _BOX_RE.finditer(answer)
    ]
    out: List[Point] = []
    for m in _POINT_RE.finditer(answer):
        s, e = m.span()
        if any(s < ie and is_ < e for (is_, ie) in consumed_intervals):
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
    anything usable" question, use `not (detections or points)` instead —
    that is the semantic the response's `abstained` field exposes, and it
    matches NVIDIA's eval pipeline which has no aggregate abstained
    concept (see module docstring)."""
    return _NONE_RE.search(answer) is not None


def _coord_valid(x1: int, y1: int, x2: int, y2: int) -> bool:
    return (
        0 <= x1 <= 1000
        and 0 <= y1 <= 1000
        and 0 <= x2 <= 1000
        and 0 <= y2 <= 1000
        and x1 < x2
        and y1 < y2
    )


def _make_box(
    label: Optional[str],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> Detection:
    return Detection(
        label=label,
        bbox_norm=[x1, y1, x2, y2],
        bbox_px=[
            round(x1 / 1000.0 * image_width, 2),
            round(y1 / 1000.0 * image_height, 2),
            round(x2 / 1000.0 * image_width, 2),
            round(y2 / 1000.0 * image_height, 2),
        ],
    )
