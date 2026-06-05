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
The [640, 2560] range is set at the training call site —
NVlabs/Eagle's `Embodied/eaglevl/train/locany_finetune_magi_stream.py:485`
(`min_long_edge=640, max_long_edge=2560, augment_prob=0.5`), which overrides
the `max_long_edge=2048` default in `Embodied/eaglevl/train/augmentation.py`;
the resize itself (uniform random long edge, Lanczos) is `augmentation.py:81,43`.

## What it does well (in order)

The model supports seven canonical task templates (detection, single /
multi phrase grounding, text grounding, scene-text detection, GUI box,
GUI point / pointing). A client never types a prompt string — it sends a
typed `request` (a sum over those seven tasks; see
[`docs/CLIENT_PROTOCOL.md`](./CLIENT_PROTOCOL.md)) and the server
**compiles** it into the exact trained prompt. **The single source of
truth for the template literals is
[`worker/prompts.py`](../worker/prompts.py)** — verbatim NVIDIA forms
with the trained `matches` / `match` asymmetry and the `</c>` category
separator. The Rust compiler at
[`rust_server/src/prompt_validator.rs`](../rust_server/src/prompt_validator.rs)
mirrors those constants byte-for-byte; a boot-time check fails the
container if the two ever drift.

Because the client only supplies typed slot values, the trained
scaffolding (the `matches` / `match` asymmetry, the `</c>` separator, the
trailing `.`) is never the client's to get wrong — it is assembled
server-side, so paraphrasing the template out of distribution is
structurally impossible. Clients either construct a typed `request`
directly, or lift a ready-made one from `/v1/capabilities.preset_prompts`
(now typed `{label, request, generation_mode}` objects). The same
`/v1/capabilities` response carries `prompt_templates_reference_url`
pointing at `worker/prompts.py`, and every per-frame slot-rejection
diagnostic includes that URL so the client always knows where to look.

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
`models/LocateAnything-3B/modeling_locateanything.py:347-353` and
`models/LocateAnything-3B/generate_utils.py`.

* **fast** — Multi-Token Prediction (MTP) only ("Parallel Box Decoding"
  in NVIDIA's README; "SDLM" block-diffusion in the code). Predicts a
  6-slot `<box>x1 x2 y1 y2</box>` block in one forward pass through the
  single shared `lm_head` (no draft heads, no speculative verification)
  and keeps it; never falls back to autoregressive. **15.3 blocks/sec —
  ~3.6× faster than slow.**
* **hybrid** (default) — MTP first; if a predicted box block is malformed
  (coords don't cleanly close) switch to AR for that block's coordinates,
  then switch back to MTP after `</box>`. **12.7 blocks/sec — ~2.9×
  faster than slow.** Best balance.
* **slow** — pure AR, the way training was supervised. **4.3 blocks/sec.**
  Highest accuracy on hard / dense / tiny / text cases.

> The decode is **lossy** parallel block prediction, NOT lossless
> speculative decoding — the code has no verify-against-the-base-model
> step, so the chosen mode genuinely changes the emitted tokens (hence the
> task-dependent F1 deltas below); it is not merely a speed knob.

### NVIDIA paper Table 12 — per-task mode metrics

The per-mode delta is task-dependent and sometimes very large. Quoted
verbatim from the paper's **Appendix C.2** ("Comprehensive Performance
Across Decoding Modes"), Table 12. Metric is **F1@mIoU** unless noted —
ScreenSpot-Pro is top-1 **Accuracy**, the Pointing block is **F1@Point**.
COCO and LVIS were evaluated at short-side 840 px; every other benchmark
at its original resolution (paper §C.5).

| Task                            | Metric   | fast | hybrid | **slow** |
|---------------------------------|----------|------|--------|----------|
| COCO (detection)                | F1@mIoU  | 52.2 | 54.7   | **55.1** |
| LVIS (open-vocab detection)     | F1@mIoU  | 47.0 | 50.7   | **52.6** |
| Dense200 (dense detection)      | F1@mIoU  | 46.8 | 61.3   | **61.5** |
| **VisDrone (tiny / sky-like)**  | F1@mIoU  | 34.4 | 39.8   | **40.2** |
| OCR HierText                    | F1@mIoU  | 28.8 | 29.1   | **43.2** |
| OCR ICDAR2015                   | F1@mIoU  | 26.6 | 26.4   | **27.3** |
| OCR TotalText                   | F1@mIoU  | 44.4 | 44.6   | **47.5** |
| **OCR SROIE**                   | F1@mIoU  | 38.8 | 39.3   | **64.4** |
| Layout DocLayNet                | F1@mIoU  | 67.2 | 77.7   | **80.4** |
| Layout M6Doc                    | F1@mIoU  | 64.1 | **70.5** | 69.7   |
| GUI ScreenSpot-Pro              | Accuracy | 59.7 | 60.3   | **60.5** |
| HumanRef (people referring)     | F1@mIoU  | 66.8 | 78.5   | **79.1** |
| RefCOCOg val (referring)        | F1@mIoU  | 70.8 | **73.4** | 72.4   |
| RefCOCOg test (referring)       | F1@mIoU  | 72.5 | **74.8** | 73.8   |
| Pointing COCO                   | F1@Point | 83.1 | 83.9   | **84.8** |
| Pointing LVIS                   | F1@Point | 74.4 | 76.6   | **76.9** |
| Pointing Dense200               | F1@Point | **89.4** | 87.6 | 88.3   |
| **Pointing VisDrone**           | F1@Point | 58.1 | 60.4   | **61.3** |

**Recommendations baked from this table:**

- **OCR / dense text (HierText, SROIE)**: use `slow`. SROIE is **+25.1 F1**
  over hybrid (64.4 vs 39.3) — the largest mode-delta in the whole
  benchmark; HierText is +14.1. (TotalText +2.9, ICDAR2015 ~+1.)
- **Tiny / distant objects (drones, VisDrone)**: use `slow`. +5.8 F1 over
  fast on detection (+0.4 over hybrid); +3.2 over fast on pointing.
- **Layout (DocLayNet)**: `slow` buys +2.7 over hybrid for fine layout;
  M6Doc is the exception (hybrid 70.5 > slow 69.7).
- **Detection (COCO/LVIS/Dense200)**: `hybrid` is the right live-video
  default — but note open-vocab LVIS buys **+1.9 F1** at `slow`
  (52.6 vs 50.7) when accuracy outweighs throughput.
- **Referring (RefCOCOg, HumanRef)**: `hybrid` — it is the *best* mode on
  both RefCOCOg splits (beats slow) and within ~0.6 on HumanRef.
- **GUI grounding (ScreenSpot-Pro)**: the three modes are within ~0.8;
  `slow` is marginally best and `fast` marginally **worst**, so `hybrid`
  is the safe middle. (A prior version of this table inverted this row and
  claimed `fast` was best — a transcription error, corrected against §C.2.)

`fast` collapses on **dense** content (Dense200 46.8 vs hybrid 61.3, a
14.5-pt cliff; all OCR), so it is never the right pick for this project's
detection/OCR workloads — only mode-insensitive GUI / easy-content
pointing tolerate it.

Every Frame's `generation_mode` field is required per request — the
server applies no default. Clients should pick based on the task per
the recommendations above; `hybrid` is the safe choice when the
workload's accuracy/throughput trade is unclear.

### A note on the attention backend

The F1 numbers above were measured by NVIDIA on H100 with the trained
`attn_implementation='magi'` path. This server runs `sdpa` + the
PyTorch memory-efficient backend on sm_120 because MagiAttention's
FA4-class cutlass kernels are not buildable on consumer Blackwell —
see [`docs/PINNED_VERSIONS.md`](./PINNED_VERSIONS.md) §`flash-attn
(source build, sm_120 only)` for the structural reason and
[`worker/inference.py`](../worker/inference.py) lines 16-46 for the
mask-equivalence argument. The two paths are INDEPENDENTLY-written mask
constructors (magi's `build_magi_ranges` vs the sdpa
`update_causal_mask_for_one_gen_window_2d`, dispatched at
`modeling_qwen2.py:1321-1335` on `_attn_implementation`); they produce
**identical attention support** — the same attend / don't-attend pattern
for every (query, key) pair — on **every (q_len, kv_len) regime this
server can actually reach** (prefill, steady-state MTP blocks, and AR
steps), confirmed by reconstructing both constructors from the
pinned-SHA source and diffing their boolean masks. This is a
*reachability* argument, not a closed-form proof of function-equality:
the one corner where the two diverge — `q_len == kv_len == block_size (6)`,
where the sdpa overlay's blocked column `K-B-1 == -1` negative-index-wraps
while magi guards it out — is structurally **unreachable** here (the
sequence is always image + prompt tokens ≫ 6, and that forward is
prefill/AR, which skips the overlay entirely). Within that identical
support the only difference is bf16 reduction-order ULP noise (~1e-3 abs
per attention call; it can perturb a softmax value but cannot flip an
attend decision); the audit bound is < 1 % of generated coord tokens
shifting by ≤ 1 quantization step (≤ 2.24 px at the 2240 px image cap).
So treat the table as accurate to ≈ ±0.2 F1, not as
attention-backend-identical to your own measurements.

The MoonViT vision encoder runs its trained **FlashAttention-2** kernel,
NOT sdpa: unlike the LLM's magi kernel, MoonViT's FA2 *is* buildable on
sm_120, and `worker/inference.py`'s `_force_vit_flash_attn_2` pins it
there. The sdpa mem-eff override for the ViT
(`_patch_vit_sdpa_to_mem_efficient` — it rebuilds the encoder's
sdpa_attention with bf16 additive masks + 4D tensors) is a **dormant
fallback** on this deployment, faithful but off the happy path. The
numerical-equivalence figures that follow therefore validate that
fallback, not the FA2 encoder the server actually runs: max absolute
delta **6e-5** vs the upstream math backend across grids from 2×2 to
160×160 patches and aspect ratios from 1:70 to 70:1 — below the bf16 ULP
at the encoder's output magnitude range. Verified in-container on
synthetic 2240×280 panorama, 280×2240 portrait, and 64×64 tiny inputs;
10-trial stability on the 16:1 panorama showed a y-stdev of 2.9 pixels.

## Generation parameters (used by every benchmark in the paper)

From NVlabs/Eagle's `Embodied/evaluation/inference_compat.py:42-68`:

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

The model emits `<box>None</box>` (capital-N — verified live; NVIDIA's
`DATA_PREPARATION.md:143` shows lowercase `none` but the trained token
id 4064 decodes to capital `None`) as a per-category abstention
placeholder. The training corpus has 22 M explicit negative queries
(16 % of the SFT data). Our parser's box/point regexes are numeric-only
— same shape as NVIDIA's eval at
`Embodied/evaluation/inference_grounding_ddp.py:282-300` — so
`<box>None</box>` silently does not match and is dropped.

The answer space is a strict **sum**: a frame's reply is exactly one of
`boxes` / `points` / `abstained` / `error` (see
[`docs/CLIENT_PROTOCOL.md`](./CLIENT_PROTOCOL.md)). **`abstained` is a
VARIANT, not a boolean** — it is the reply the server returns iff the
model produced NO parseable geometry at all (no boxes and no points):
every `<box>None</box>` or empty output. It deliberately collapses "all
categories abstained" with "model produced unparseable output"; both are
"nothing to render" from the client's perspective. NVIDIA's own eval
pipeline has no aggregate-abstention concept either —
`Embodied/evaluation/metrics/other_metric.py:140-156` only tracks
per-category None.

Off-contract output is handled **per-item**, orthogonally to the
abstained variant: geometry in the wrong shape for the task (a point on a
box-shaped task, or vice-versa) and any box/point with no non-empty
`<ref>` label are dropped from the returned array but **counted** in the
`deviations_dropped` metadata field, while the valid co-emitted geometry
is still returned in the `boxes`/`points` variant. So a model that DID
localize the object — just emitted some surplus off-shape tokens
alongside — is never misreported as having abstained. The decision is
made on the parse, never from a substring scan over `raw_text` (a
substring scan would mis-trip on an absent-category
`<ref>X</ref><box>None</box>` triple co-occurring with real detections).
A frame where the model emitted geometry but **zero** of it was valid for
the task (all off-shape / unlabeled), or where the output was
unparseable, is an `error{code:"model_deviation"}` — not `abstained`,
because the model did not actually decline; it deviated.

For per-category abstention in a multi-category detection prompt,
clients recover the absent categories as the set difference between
the request's category list and `{b.label for b in boxes}` (on a
`boxes` reply) — the `<ref>X</ref><box>None</box>` triples for those
categories are also present verbatim in `raw_text` if needed.
**`abstained` is not a calibrated confidence** — treat it as "no usable
output", not "model is confident there is nothing".

## Coordinate system

The model emits integer coords in `[0, 1000]` (**inclusive of both
endpoints** — verified against `added_tokens.json:151677..152677`
which contains 1001 discrete coord tokens `<0>` … `<1000>`).

* **Box order** is `<box><x1><y1><x2><y2></box>` — verified against
  `Embodied/document/DATA_PREPARATION.md:131` and the
  `Embodied/evaluation/inference_*.py` parsers. A misleading comment
  in `models/LocateAnything-3B/generate_utils.py:297` calls positions
  "`x1,x2,y1,y2`" — that's positional-naming, not coordinate-semantic;
  the model emits x1,y1,x2,y2.
* **Point order** is `<box><x><y></box>` (note: only two coords; the
  `<box>` delimiter is reused for points — there is no separate
  `<point>` token).
* **Origin** is the top-left of the image at PIL `(0,0)`.
* **Scale** is per-axis, not uniform. The image processor optionally
  downscales the whole image uniformly by `sqrt(25600 / total_patches)`
  (aspect preserved) when the patch budget is exceeded, then
  **anamorphically resizes** it (via `image.resize(...)`, not a pad) so each
  side is a multiple of 28 px — the independent ceil-to-28 makes the x and y
  factors differ slightly. The coordinate round-trip is exact regardless,
  because each axis is normalized INDEPENDENTLY: `coord/1000 × source_w` for
  x and `coord/1000 × source_h` for y recover the source-relative pixel per
  axis. Multiplying by the source dimensions is the canonical convention and
  that's what the server does in `bbox_px`.
* **Quantization granularity** per axis = `image_dim / 1000`. For a
  2240 px square image, 1 coord-token = 2.24 px. For 1920 px wide,
  1 coord-token = 1.92 px.

Each box (under the `boxes` variant) and each point (under the `points`
variant) carries a required `label` plus both coordinate forms:

- `bbox_norm: [x1, y1, x2, y2]` (boxes) / `point_norm: [x, y]` (points) —
  canonical integer coords in `[0,1000]`: each coord clamped to the grid
  and, for boxes, corners min/max-sorted (`x1<=x2`, `y1<=y2`), matching
  NVIDIA's eval. The verbatim token-order emission is in `raw_text`.
- `bbox_px: [x1, y1, x2, y2]` (boxes) / `point_px: [x, y]` (points) —
  pixels relative to the **source** image dimensions (float, rounded to
  2 decimal places).
