# Model capabilities — what LocateAnything-3B actually does

This document is a deliberate, blunt summary of **what the model was
trained for and what it isn't trained for**. The setup serves the
model exactly as NVIDIA released it, with no fine-tuning, no LoRAs,
no second-stage detector. Every number below is verifiable in the
upstream paper, the model card, the inference scripts, or the
training scripts under [`NVlabs/Eagle/Embodied`](https://github.com/NVlabs/Eagle/tree/main/Embodied).

---

## Identity

* **Model**: `nvidia/LocateAnything-3B`
* **Paper**: *LocateAnything: Fast and High-Quality Vision-Language
  Grounding with Parallel Box Decoding* (arXiv: 2605.27365)
* **Released**: 2026-05-26
* **Largest variant**: there is only one variant — 3 B parameters
  (Qwen2.5-3B-Instruct LLM + MoonViT-SO-400M vision encoder).
* **License**: NVIDIA License (non-commercial use only for the
  weights); Apache 2.0 for code. **You may NOT use the weights for
  commercial purposes** except via direct agreement with NVIDIA.

## What it was trained on

The total LocateAnything-Data SFT corpus is 138 M queries / 785 M
boxes / 12 M images, split roughly:

| Domain     | Share | Datasets |
|---|---|---|
| Detection  | 66.9 % | Objects365 (47.3M), OpenImages (41.0M), V3Det, DeepFashion2, PartImageNet++, BDD100K dashcam, NuImages, MOT17/20Det street CCTV, SKU110K, CrowdHuman, OWDOD, EgoObjects, PACO |
| GUI        | 16.5 % | OSAtlas (14.3M), GroundCUA, ScaleCUA, GTAGrounding, MultiUI |
| Referring  |  7.3 % | OpenImages, Object365, Unsplash, gRefCOCO, RefCOCO, RefCOCO+, RefCOCOg, Flickr30kEntities, HumanPart, HumanRef, RoboAffordance |
| OCR        |  3.6 % | BLIP3OCR (3.8M), HierText, ReCTS, IDLOCR, Art, COCO_Text V2, ICDAR2013/2015, LSVT, TextOCR, SROIE, WildReceipt, RCTW |
| Layout     |  3.5 % | PubLayNet (3.4M), DocLayNet (1.0M), TableBank, M6Doc, TabRecSet, CDLA |
| Pointing   |  2.2 % | Object365, OpenImages, PixmoPoints, RoboAffordance |

**Augmentation**: only random resize (50 % probability, long-edge
in [640, 2560], Lanczos). No motion-blur, no Gaussian-blur, no
color/brightness jitter, no noise, no JPEG-quality augmentation.
Verified in
`/tmp/nvlabs_eagle/Embodied/eaglevl/train/augmentation.py`.

## What it does well (in order)

The model supports seven canonical task templates (detection, single /
multi phrase grounding, text grounding, scene-text detection, GUI box,
GUI point / pointing). **The single source of truth for those templates
is [`worker/prompts.py`](../worker/prompts.py)** — verbatim NVIDIA forms
with the trained `matches` / `match` asymmetry and the `</c>` category
separator. The Rust validator at
[`rust_server/src/prompt_validator.rs`](../rust_server/src/prompt_validator.rs)
mirrors those constants byte-for-byte and enforces them at the WebSocket
edge; a boot-time check fails the container if the two ever drift.

Clients should either call the `detect_categories`, `ground_single`,
`ground_multi`, `ground_text`, `detect_text`, `ground_gui`, or `point_to`
helpers from `worker/prompts.py`, or pull pre-built well-formed prompts
from `/v1/capabilities.preset_prompts`. The same `/v1/capabilities`
response also carries `prompt_templates_reference_url` pointing at
`worker/prompts.py`, and every per-frame rejection diagnostic includes
that URL so the client always knows where to look. **Do not paraphrase**
— per-word deviations move you off the training distribution.

## What it does NOT do well

* **Tiny objects, especially against a clean sky background.** No
  aerial / sky / surveillance imagery in training. VisDrone
  F1@0.95 = 3.2 (paper Tab. 2). The drone-pointing F1@Point of 60.4
  is *better* than the box version but is still on
  drone-mounted-looking-down imagery, not ground-mounted-looking-up.
* **Motion-blurred frames.** No motion-blur augmentation in training.
* **Low-light frames.** No brightness/contrast augmentation.
* **Out-of-distribution category names.** "drone" is in the
  vocabulary (it's in Objects365 / OpenImages), but
  "FPV racing quadcopter" or "small dark dot in the sky" are
  ungrounded by training.
* **High-IoU localization in dense scenes.** Even on benchmarks where
  the average F1 is good (LVIS 50.7 mean), F1@0.95 drops sharply
  (LVIS 31.1, COCO 19.3).

## Generation modes

`fast`, `hybrid`, `slow`. Verified in
`/tmp/la3b_meta/modeling_locateanything.py:347-353` and
`/tmp/la3b_meta/generate_utils.py`.

* **fast** — Multi-Token Prediction (MTP) only. Predicts the 6-token
  `<box>...<...>...</box>` block in a single forward pass and never
  falls back to autoregressive decoding. ~3x faster than slow.
* **hybrid** (default) — MTP first; if the predicted block doesn't
  match the box pattern, switch to AR for the malformed coords,
  switch back to MTP after `</box>`. Best balance.
* **slow** — pure AR, the way training was supervised. Highest
  accuracy on hard / dense / tiny cases.

### NVIDIA paper Table 12 — per-task mode F1

The F1@mIoU delta between modes is task-dependent and sometimes very
large. Quoted directly from paper §C.4 (verified by the deep-audit
subagent):

| Task                            | fast | hybrid | **slow** | Pick |
|---------------------------------|------|--------|----------|------|
| LVIS (open-vocab detection)     | ~50  | 50.7   | 50.9     | hybrid |
| COCO (detection)                | 54.0 | 54.7   | 54.9     | hybrid |
| Dense200 (dense detection)      | 46.8 | 61.3   | **61.5** | hybrid or slow |
| **VisDrone (tiny / sky-like)**  | 34.4 | 39.8   | **40.2** | **slow** |
| RefCOCOg test (referring)       | 75.8 | 77.6   | 77.5     | hybrid |
| HumanRef (people referring)     | 76.4 | 78.7   | 78.9     | hybrid |
| ScreenSpot-Pro (GUI point)      | 60.3 | 60.4   | 60.2     | **fast** |
| **OCR HierText**                | 28.8 | 29.1   | **43.2** | **slow** |
| **OCR SROIE**                   | 38.8 | 39.3   | **64.4** | **slow** |
| OCR ICDAR2015 / TotalText       | ~50  | ~50    | ~50      | hybrid |
| Layout DocLayNet                | 75.8 | 76.8   | 76.9     | hybrid |
| Pointing VisDrone               | 58.1 | 60.4   | **61.3** | slow |

**Recommendations baked from this table:**

- **Tiny / distant objects (drones, surveillance, VisDrone-style)**:
  use `slow`. Up to **+5 F1** over fast on this regime.
- **OCR (HierText, SROIE)**: use `slow`. Up to **+25 F1** over hybrid
  on SROIE — the largest mode-delta in the entire benchmark.
- **GUI grounding**: `fast` is fine — the three modes are within 0.2
  F1 of each other and fast is 3× cheaper.
- **Detection / referring / layout** (the bulk of normal use):
  `hybrid` is the right default — the slow upside is sub-1-F1.

The server's `LA_GEN_MODE=hybrid` default reflects "what to use when
you don't know" — but every Frame's `generation_mode` field is
required per request, so clients should select based on the task.

## Generation parameters (used by every benchmark in the paper)

From `/tmp/nvlabs_eagle/Embodied/evaluation/inference_compat.py:42-68`:

```python
do_sample=True,
temperature=0.7,
top_p=0.9,
repetition_penalty=1.1,
n_future_tokens=6,
use_cache=True,
```

This server uses exactly these values, baked in as Docker `ENV`
variables (`LA_GEN_TEMPERATURE`, `LA_GEN_TOP_P`, etc.). The Docker
image refuses to start if any of them is unset.

`max_new_tokens` is set to **8192** (the README's recommended value),
capped by the tokenizer's `model_max_length=16384` minus the input
length.

## Time scale invariance

Single-image inference is **stateless across calls**. The Worker
loads weights once; each `model.generate(...)` call builds a fresh
KV cache for its input, runs to EOS, and returns. No timestamp,
frame index, or time delta is ever passed to the model.

Consequence: feeding the model 1 FPS or 30 FPS makes no difference to
the model's output per frame. Live throughput is **purely GPU-bound**.
The server measures its sustainable FPS at boot
(`worker/calibration.py`) and exposes it via `/v1/capabilities`. The
server **does not drop frames** based on wall-clock — backpressure
via TCP flow control is the only mechanism.

The model also has a *native* video path (`processor.process_vision_info`
accepts `{"type":"video", ...}` and stitches up to 64 frames at
`fps=2.0` into a single context). We don't use that path for live
operation — it's a batched, finite-clip mode at training-time fps,
not a streaming mode.

## Abstention

The model can emit `<box>none</box>` to signal "no detection". The
training corpus has 22 M explicit negative queries (16 % of the SFT
data). The server parses this signal (`worker/parsing.py:has_abstention`)
and exposes `"abstained": true` in the response. **It is not a
calibrated confidence** — treat it as a hint, not a guarantee.

## Coordinate system

The model emits integer coords in `[0, 1000]` (**inclusive of both
endpoints** — verified against `added_tokens.json:151677..152677`
which contains 1001 discrete coord tokens `<0>` … `<1000>`).

* **Box order** is `<box><x1><y1><x2><y2></box>` — verified against
  `Embodied/document/DATA_PREPARATION.md:131` and the
  `Embodied/evaluation/inference_*.py` parsers. A misleading comment
  in `/tmp/la3b_meta/generate_utils.py:297` calls positions
  "`x1,x2,y1,y2`" — that's positional-naming, not coordinate-semantic;
  the model emits x1,y1,x2,y2.
* **Point order** is `<box><x><y></box>` (note: only two coords; the
  `<box>` delimiter is reused for points — there is no separate
  `<point>` token).
* **Origin** is the top-left of the image at PIL `(0,0)`.
* **Scale** is uniform (same factor in x and y). The image processor
  scales the whole image by `sqrt(25600 / total_patches)` when the
  patch budget is exceeded, then pads to a multiple of 28 px.
  Because the scale is uniform, `coord/1000 × source_w` and
  `coord/1000 × resized_w` give the same source-relative pixel —
  multiplying by the source dimensions is the canonical convention
  and that's what the server does in `bbox_px`.
* **Quantization granularity** per axis = `image_dim / 1000`. For a
  2240 px square image, 1 coord-token = 2.24 px. For 1920 px wide,
  1 coord-token = 1.92 px.

The server returns:

- `bbox_norm: [x1, y1, x2, y2]` — model output unchanged (integers in `[0,1000]`)
- `bbox_px:   [x1, y1, x2, y2]` — pixels relative to the **source**
  image dimensions (float, rounded to 2 decimal places).
