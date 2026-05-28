# Pixel-to-token math

Every claim in this file is verifiable against the model files in
[`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B):
specifically `image_processing_locateanything.py`,
`preprocessor_config.json`, and `config.json`. If a number here
disagrees with the upstream model, treat the upstream as authoritative
and open an issue.

---

## The geometry

* **ViT patch size**: 14 pixels. Verified in
  `config.json` (`vision_config.patch_size`) and
  `preprocessor_config.json` (`patch_size`).
* **2×2 patch merger**: every 2×2 block of ViT patches is concatenated
  along channel and projected through a `mlp1` (LayerNorm → Linear →
  GELU → Linear) into the LLM's hidden dimension. So one **LLM
  token** spans a **28 × 28 px** region of the resized image.
* **In-token limit** (the cap on total ViT patches per image):
  **25,600** — read from `preprocessor_config.json:in_token_limit`.
* **Max LLM tokens per image** (after 2×2 merger): **6,400**.
* **Positional embedding limit per side**: 512 patches. Combined with
  the 25,600-patch cap, the realistic per-side max is whatever leaves
  the product below 25,600.
* **Coord vocabulary**: the model has dedicated tokens `<0>` … `<1000>`
  (1001 tokens) for spatial coordinates. Output positions are
  quantized into this grid. For a 2,240-px image, **1 coord token = 2.24
  px**; for a 1,920-px image, **1 coord token = 1.92 px**.

## What the image processor actually does (verified against code)

```
rescale(w, h):
    n_patches = (w // 14) * (h // 14)
    if n_patches > 25_600:
        scale = sqrt(25_600 / n_patches)
        w, h = round(w * scale), round(h * scale)
    # pad up to multiples of 28 (= 14 * 2)
    dst_w = ceil(w / 28) * 28
    dst_h = ceil(h / 28) * 28
    if dst_w // 14 >= 512 or dst_h // 14 >= 512:
        raise ValueError("Exceed pos emb")
    return dst_w, dst_h
```

Practical consequences:

| Source resolution    | Source patches | After resize+pad     | LLM tokens | scale  | 1 LLM token in src px |
|---|---|---|---|---|---|
| 1920 × 1080 (1080p)  | 10,549 ≤ 25.6k → no rescale | 1932 × 1092 (138×78 patches) | 2,691  | 1.006  | ≈ 27.83 |
| 2560 × 1440 (1440p)  | 18,775 ≤ 25.6k → no rescale | 2576 × 1456 (184×104 patches)| 4,784  | 1.006  | ≈ 27.83 |
| 3840 × 2160 (4K)     | 42,196 > 25.6k → ↓0.7802    | 2996 × 1708 (214×122 patches)| 6,527  | 0.7802 | ≈ 35.89 |
| 2240 × 2240 (square) | 25,600 = cap → no rescale   | 2240 × 2240 (160×160 patches)| 6,400  | 1.000  | 28.00   |

Note that the 4K case lands just over the 6,400-LLM-token cap (6,527),
because the upstream preprocessor rescales by `sqrt(25600/total_patches)`
and then pads up to multiples of 28 — the pad step can push the
post-resize patch count slightly above 25,600. The model still
accepts this (the only hard ceiling is the per-side 512-patch
positional-embedding limit, which 214×122 is well under).

## What this means for object size

If your target subtends N pixels in the **source** image, the model's
"view" of it is roughly:

* `≈ N × scale` pixels in the model's input,
* `≈ N × scale / 28` LLM tokens.

A target that is **less than 28 / scale source pixels** in either
direction is *sub-token* — the model has effectively no spatial
information about it.

For FPV-drone detection from a 4K ground-facing camera:

* Drone subtends **5 px** in source → 0.14 LLM tokens. **Hopeless.**
* Drone subtends **30 px** in source → 0.84 LLM tokens. **Marginal.**
* Drone subtends **100 px** in source → 2.8 LLM tokens. **Likely
  detectable** by the model's open-vocab grounding, assuming the
  drone is on a clean background.

This is also the basis of the model's reported VisDrone numbers
(F1@0.95 = 3.2, mean 39.8) — tight-IoU localization at small object
sizes is broken for this class of architecture, not just this model.

## Tiling rescues some recall — but the server does NOT do it for you

If you must detect tiny targets in a 4K source, run the model on
overlapping tiles instead of the whole frame:

* A 4 × 3 grid with 15 % overlap → twelve ~1,300 × 850-px tiles.
* Each tile fits in the 25,600-patch budget at native resolution
  (≈ 5,500 patches), so the rescale step is a no-op per tile.
* A 30-px drone in the source remains 30 px in the tile and is now
  ~1 LLM token (28 px) — borderline detectable instead of sub-token.
* External NMS merges duplicates from the overlap bands.

**This is a client-side responsibility.** The server's API is single-
frame, single-prompt — one Frame in, one Result out. Tiling would
multiply inference time by `rows × cols` and would mean either an
opt-in protocol field (which violates our single-API-surface rule)
or making every request pay the tiling tax. The right architectural
place for the tiling loop is in the client — it knows the camera
resolution, the desired tile geometry, and what to do with the
N separate Results.

The `worker/tiling.py` module contains the geometry helpers (grid
generation, IoU, NMS) ready for adoption into a downstream
tile-orchestrating client. It is not invoked by the server itself
and is exempt from the wire protocol.

## How the server exposes this

`GET /v1/info` returns the resize plan for several reference
resolutions:

```json
{
  "pixel_token_examples": {
    "1080p":       { "dst_w": 1932, "dst_h": 1092, "n_llm_tokens": 2691, ... },
    "1440p":       { "dst_w": 2576, "dst_h": 1456, "n_llm_tokens": 4784, ... },
    "4K":          { "dst_w": 2996, "dst_h": 1708, "n_llm_tokens": 6527, ... },
    "square_2240": { "dst_w": 2240, "dst_h": 2240, "n_llm_tokens": 6400, ... }
  }
}
```

Clients can use this to decide whether to tile.
