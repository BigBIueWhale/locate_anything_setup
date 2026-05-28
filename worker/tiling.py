"""
External tiling for tiny-object recall.

The LocateAnything repo does NOT provide tiling. The model's effective
spatial resolution is ~28 px per LLM token after the 2×2 patch merger,
so any object smaller than ~28 input pixels has sub-token spatial
information after resize-to-token-budget. For the user's drone-detection
use case this is the binding constraint.

External tiling: chop the source frame into an NxM overlapping grid,
run the model on each tile (at the model's max usable resolution per
tile), parse boxes in tile-local coords, transform back to global
coords, then apply NMS to merge duplicates from overlap regions.

This module implements ONLY the geometric helpers — no model touch.
The orchestration (loop, NMS, etc.) is intentionally NOT in
la_worker.py: the server's API is single-frame, single-prompt,
and tiling is a client-side responsibility because the client
knows its camera geometry and how to merge per-tile results.
See docs/DRONE_DETECTION.md and docs/PIXEL_TO_TOKEN_MATH.md for the
recommended client-side pattern. The helpers here are ready for
adoption into such a client.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Tile:
    """A rectangular crop in the source image's pixel coordinate system."""
    row: int
    col: int
    x0: int  # inclusive
    y0: int  # inclusive
    x1: int  # exclusive
    y1: int  # exclusive

    @property
    def width(self) -> int:  return self.x1 - self.x0
    @property
    def height(self) -> int: return self.y1 - self.y0


def grid(
    image_width: int,
    image_height: int,
    rows: int,
    cols: int,
    overlap_ratio: float = 0.15,
) -> List[Tile]:
    """
    Build an `rows × cols` grid of tiles covering the source image with
    `overlap_ratio` overlap between neighbours. Overlap is measured as
    a fraction of tile dimension and applied symmetrically to both edges
    where applicable.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be ≥ 1")
    if not (0.0 <= overlap_ratio < 0.5):
        raise ValueError("overlap_ratio must be in [0.0, 0.5)")

    tile_w_base = image_width / cols
    tile_h_base = image_height / rows
    pad_x = int(round(tile_w_base * overlap_ratio))
    pad_y = int(round(tile_h_base * overlap_ratio))

    out: List[Tile] = []
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, int(round(c * tile_w_base)) - pad_x)
            y0 = max(0, int(round(r * tile_h_base)) - pad_y)
            x1 = min(image_width, int(round((c + 1) * tile_w_base)) + pad_x)
            y1 = min(image_height, int(round((r + 1) * tile_h_base)) + pad_y)
            out.append(Tile(row=r, col=c, x0=x0, y0=y0, x1=x1, y1=y1))
    return out


def remap_box_to_global(
    bbox_px_local: list,  # [x1, y1, x2, y2] in tile-local pixels
    tile: Tile,
) -> list:
    """Transform a tile-local pixel bbox back into global source image pixels."""
    x1, y1, x2, y2 = bbox_px_local
    return [
        tile.x0 + x1,
        tile.y0 + y1,
        tile.x0 + x2,
        tile.y0 + y2,
    ]


def iou(a: list, b: list) -> float:
    """Standard IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(
    boxes: List[dict],
    iou_threshold: float = 0.5,
    label_aware: bool = True,
) -> List[dict]:
    """
    Non-maximum suppression over a list of detections (dicts with
    `bbox_px` and optionally `label`). The model emits no calibrated
    confidence, so we use box AREA as the proxy score — larger boxes
    'win' overlap conflicts. This is a deliberate, documented choice:
    for tiny-object detection, a larger box absorbing a tiny one is
    a reasonable failure mode.
    """
    if not boxes:
        return []
    # Sort by area descending.
    def area(b):
        x1, y1, x2, y2 = b["bbox_px"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    sorted_boxes = sorted(boxes, key=area, reverse=True)
    keep: List[dict] = []
    for b in sorted_boxes:
        dup = False
        for k in keep:
            if label_aware and (b.get("label") != k.get("label")):
                continue
            if iou(b["bbox_px"], k["bbox_px"]) >= iou_threshold:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep
