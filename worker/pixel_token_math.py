"""
Pixel-to-token geometry for LocateAnything-3B.

Verified against `models/LocateAnything-3B/image_processing_locateanything.py`
and `preprocessor_config.json` (see docs/PIXEL_TO_TOKEN_MATH.md for derivation).

Key constants — DO NOT change unless re-verifying against the model's
preprocessor_config.json upstream.
"""

from __future__ import annotations
from dataclasses import dataclass
import math

# MoonViT patch size (pixels).
PATCH_SIZE = 14
# 2×2 patch merger feeding the LLM — every 2×2 block of ViT patches becomes
# 1 LLM token. Therefore 1 LLM token spans a 28×28 px region.
MERGE_KERNEL = 2
LLM_TOKEN_PX = PATCH_SIZE * MERGE_KERNEL  # 28
# Total ViT patches allowed per image, per preprocessor_config.json.
IN_TOKEN_LIMIT = 25600
# Effective max LLM tokens per image after merger.
MAX_LLM_TOKENS_PER_IMAGE = IN_TOKEN_LIMIT // (MERGE_KERNEL * MERGE_KERNEL)  # 6400
# Hard ceiling per side in patches (modeling_vit RoPE2D limit).
MAX_PATCHES_PER_SIDE = 512


@dataclass(frozen=True)
class ImageResize:
    """The fully-resolved resize plan for one input image."""
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    n_patches_w: int
    n_patches_h: int
    n_llm_tokens: int
    scale: float

    def min_resolvable_object_px_src(self) -> float:
        """In source-image pixels, the side length of an object that occupies
        exactly one LLM token after resize+merger. Objects smaller than this
        are effectively sub-token and very hard to ground."""
        return LLM_TOKEN_PX / self.scale


def plan_resize(width: int, height: int) -> ImageResize:
    """
    Compute the rescale+pad plan the LocateAnything image processor will
    apply to an input of (width, height).

    Mirrors `LocateAnythingImageProcessor.preprocess` exactly:
      - If (w//14)*(h//14) > 25600, downscale by sqrt(25600/total_patches).
      - Pad up so both H and W are divisible by 28 (= patch*merge).
      - Raise if w//14 or h//14 ≥ 512 (positional embedding limit).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image dims must be positive, got ({width}, {height})")

    w, h = width, height
    total_patches = (w // PATCH_SIZE) * (h // PATCH_SIZE)
    if total_patches > IN_TOKEN_LIMIT:
        scale = math.sqrt(IN_TOKEN_LIMIT / total_patches)
        w = max(int(round(w * scale)), PATCH_SIZE)
        h = max(int(round(h * scale)), PATCH_SIZE)
    # Pad up to multiple of 28.
    grid = PATCH_SIZE * MERGE_KERNEL
    dst_w = ((w + grid - 1) // grid) * grid
    dst_h = ((h + grid - 1) // grid) * grid

    n_patches_w = dst_w // PATCH_SIZE
    n_patches_h = dst_h // PATCH_SIZE
    if n_patches_w >= MAX_PATCHES_PER_SIDE or n_patches_h >= MAX_PATCHES_PER_SIDE:
        raise ValueError(
            f"image {width}x{height} → grid {n_patches_w}x{n_patches_h} "
            f"exceeds MAX_PATCHES_PER_SIDE={MAX_PATCHES_PER_SIDE} after resize"
        )
    n_llm_tokens = (n_patches_w * n_patches_h) // (MERGE_KERNEL * MERGE_KERNEL)
    scale = dst_w / width  # uniform x and y (aspect preserved before pad)
    return ImageResize(
        src_w=width,
        src_h=height,
        dst_w=dst_w,
        dst_h=dst_h,
        n_patches_w=n_patches_w,
        n_patches_h=n_patches_h,
        n_llm_tokens=n_llm_tokens,
        scale=scale,
    )


def summarize(width: int, height: int) -> dict:
    """JSON-serializable summary, for /v1/info."""
    plan = plan_resize(width, height)
    return {
        "src_w":              plan.src_w,
        "src_h":              plan.src_h,
        "dst_w":              plan.dst_w,
        "dst_h":              plan.dst_h,
        "n_patches_w":        plan.n_patches_w,
        "n_patches_h":        plan.n_patches_h,
        "n_llm_tokens":       plan.n_llm_tokens,
        "scale":              plan.scale,
        "min_resolvable_object_px_src": round(plan.min_resolvable_object_px_src(), 2),
        "patch_px":           PATCH_SIZE,
        "llm_token_px_post_merger": LLM_TOKEN_PX,
        "in_token_limit":     IN_TOKEN_LIMIT,
        "max_llm_tokens_per_image": MAX_LLM_TOKENS_PER_IMAGE,
    }
