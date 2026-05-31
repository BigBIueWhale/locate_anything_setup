# Drone detection — capabilities, limits, and honest assessment

This document is here because the project was set up with the
intent of building **ground-based FPV-drone (first-person-view drone) early-warning** on top
of LocateAnything-3B. After verifying the model's training data,
architecture, and benchmarks, the honest answer is:

> **LocateAnything-3B is the wrong primary tool for this job.**
> Use it as a downstream verifier on ROIs cropped by a specialized
> detector, not as the primary detector itself.

The full evidence is in
[`MODEL_CAPABILITIES.md`](./MODEL_CAPABILITIES.md) and
[`PIXEL_TO_TOKEN_MATH.md`](./PIXEL_TO_TOKEN_MATH.md). The short
version follows.

---

## The five reasons

1. **No relevant training data.** The LocateAnything-Data corpus
   (138 M queries) contains zero aerial / sky / ground-pointing
   surveillance imagery. The closest is BDD100K dashcam — looking
   *forward*, not *up*. No drone-mounted top-down, no
   ground-pointing telephoto, no anti-UAV.

2. **Effective spatial floor ≈ 28 source px.** With the 14-px ViT
   patch and 2×2 merger, 1 LLM token = 28 × 28 px of the (resized)
   input. A 4K-source drone smaller than ~36 source px (after the
   automatic rescale to fit 25,600 patches) is *sub-token* — the
   model has effectively zero spatial information about it.

3. **VisDrone benchmark is catastrophic at high IoU.** F1@0.95 = 3.2.
   Mean F1 = 39.8 — *only* +1.3 over Grounding-DINO-Swin-T from
   2023. A 3-B-parameter VLM is not beating a 2023 specialized
   detector on this kind of task.

4. **No augmentation for the input distribution.** Training was
   resize-only. No motion blur, no noise, no brightness/contrast
   jitter, no low-light. Real FPV-drone footage has all of these.

5. **No calibrated confidence.** The model emits boxes or
   `<box>none</box>`. There is no per-box probability the user can
   threshold against. Specialized detectors emit `objectness × class`
   scores that are calibrated for downstream tracking; this model
   doesn't.

## What WILL work, qualitatively

* **Detecting a drone that subtends ≥ 60 px in the source image,
  against a clean sky, in good light, with no motion blur.**
  E.g., a quadcopter at 10–30 m through a 50 mm lens on a 1080p
  camera in daylight. Use `prompts.point_to("drone in the sky")`
  (see [`worker/prompts.py`](../worker/prompts.py)) in `slow` mode.

* **Verifying / classifying ROIs already cropped by another
  detector.** When the drone occupies a significant fraction of
  the crop, LocateAnything's open-vocab strength kicks in.
  Use `prompts.ground_single("a quadcopter drone")` on the ROI.

* **Demonstrating the model on household / consumer objects.**
  The HF Space's default prompts (`book`, `sweet`, `person`,
  `text`, `car, bus, person, potted plant`) and the
  `worker.prompts.HOUSEHOLD_PROMPTS` bundle all work well — that's
  what the model was actually trained for.

## What WILL NOT work

* Detecting a drone smaller than ~30 source px in a 4K frame.
* Detecting against textured backgrounds (clouds, foliage,
  power lines) — high false-positive rate.
* Real-time at 30+ fps — see "Throughput" below.

## If you must use LocateAnything as the *primary* detector

The server's API is single-frame: one Frame in, one Result out. There
is no server-side tiling mode — that decision rightly belongs to the
client, which knows its camera geometry and what to do with multiple
per-tile Results.

Recommended client-side approach for a 4K camera:

* Chop the source frame into a **4 × 3** grid of ~1,300×850 tiles with
  **15 %** overlap.
* Per tile, JPEG-encode, send a Frame with `generation_mode = "slow"`
  and prompt built via `prompts.ground_multi("a small drone in the sky")`
  (see [`worker/prompts.py`](../worker/prompts.py)).
* Parse each Result's `detections[].bbox_px` and translate back into
  source-image coords using the tile's offset.
* Apply NMS across all tiles to merge duplicates from the overlap
  bands.
* **Temporally smooth** over 3-5 frames before declaring a track —
  the model has no temporal model itself.

This multiplies inference cost by 12 per frame. At the measured
single-frame `slow`-mode throughput of ~1.07 FPS on 1080p drone
content (table below), that's ~0.09 FPS end-to-end per 4K source
frame. The geometry helpers in
[`worker/tiling.py`](../worker/tiling.py) (grid generation, IoU, NMS)
are ready for adoption into such a client.

## Throughput on RTX 5090

Measured in-container with 315 inferences = 3 real drone JPEGs
(1080p-class, within the 25,600-patch budget so the preprocessor does
no rescale) × the seven canonical drone prompts
(`prompts.DRONE_PROMPTS_RANKED`) × all three generation modes × 5
trials, using the trained sampling parameters (`temperature=0.7,
top_p=0.9, repetition_penalty=1.1`) and the SDPA mem-eff attention
path (MagiAttention is unbuildable on sm_120 — see
[`docs/PINNED_VERSIONS.md`](./PINNED_VERSIONS.md)).

| Mode     | mean gen_time | p95   | tokens/s | mean num_boxes | mean forward_steps |
|----------|--------------:|------:|---------:|---------------:|-------------------:|
| `fast`   |        757 ms | 1.11s |     35.8 |           1.48 |                4.3 |
| `hybrid` |        785 ms | 1.17s |     32.7 |           1.56 |                6.4 |
| `slow`   |        937 ms | 1.61s |     24.7 |           1.62 |               17.6 |

Multi-Token Prediction (MTP) cuts forward passes by ~73 % in `fast` and ~64 % in `hybrid`
relative to `slow` (mean `fast/slow=0.27`, `hybrid/slow=0.36`).
Wall-clock improvement is modest (~17 % `hybrid → slow`) because each
inference is dominated by the prefill (vision-token encoding + first
LLM forward) on a 1080p input; MTP's speedup amortizes over output
length, so the paper's 2.5× headline applies to high-density
detection (Dense200, COCO) rather than 1-3-box drone shots.

### MTP acceptance (`hybrid` mode)

Across 105 `hybrid` inferences, MTP emitted 464 multi-token blocks of
which 45 were structurally malformed and triggered a single-token AR
completion before returning to MTP. **Acceptance rate ≈ 90.3 %.** 74
of 105 inferences (70.5 %) saw zero AR fallbacks; the per-inference
failure distribution is `{0:74, 1:26, 3:1, 4:4}`. `fast` and `slow`
report zero `switch_to_ar` events by construction (`fast` never falls
back; `slow` never tries MTP).

The structural difficulty varies sharply by prompt shape (mean
`switch_to_ar` over 15 `hybrid` trials each):

| Prompt | mean | max |
|--------|-----:|----:|
| `Locate all the instances that matches the following description: drone</c>quadcopter</c>uav</c>aircraft.` | 1.60 | 4 |
| `Locate all the instances that match the following description: a small drone in the sky.`               | 0.67 | 1 |
| `Locate all the instances that matches the following description: drone.`                                | 0.33 | 1 |
| `Locate all the instances that match the following description: a flying object in the sky.`             | 0.33 | 1 |
| `Point to: drone in the sky.`                                                                            | 0.07 | 1 |
| `Locate a single instance that matches the following description: the drone.`                            | 0.00 | 0 |
| `Point to: quadcopter.`                                                                                  | 0.00 | 0 |

Multi-category detection has the most MTP structural complexity (the
output must interleave `<ref>X</ref><box>…</box>` per category in
parallel); pointing and single-instance grounding are the cleanest.
The 10 % malformed-MTP events are invisible at the response layer —
`hybrid` AR-completes them in place and the output parses cleanly —
but they DO cost the forward-pass savings: those tokens run at
`slow`-mode speed.

### Tiling math (4 × 3 grid of 4K tiles)

* `hybrid`: 12 × 785 ms ≈ 9.4 s per source frame (~0.11 fps).
* `slow`  : 12 × 937 ms ≈ 11.2 s per source frame (~0.09 fps).

The worker serializes tile inference (single-concurrent-user design
— see [`docs/OPERATIONS.md`](./OPERATIONS.md#concurrency)).

The `calibration` block returned by `GET /v1/capabilities` reports a
6-run inference cycle on `test_data/drone_sirius.jpg` with
`prompts.point_to("drone in the sky")`. That specific prompt is the
structurally cleanest of the seven canonical drone prompts (mean
`switch_to_ar=0.07`, ~10 output tokens), so the per-boot
`median_fps` (~4.8 FPS, ~210 ms on this hardware) characterises the
**best-case** drone workload — not the aggregate. Expect proportionally
lower throughput for prompts that emit more output tokens (the
throughput table above shows `hybrid: 785 ms` averaged across all 7
canonical drone prompts and 3 drone JPEGs, including multi-category
detection which is ~3-4× slower than `point_to`). The default
calibration target is overridable via `LA_CALIBRATION_IMAGE` /
`--calibration-image` (and `--calibration-prompt`) if you want a
different boot-time characterisation.

## Recommendation

The correct production architecture for drone early-warning is:

```
        ┌─ YOLOv8/v9-p2 trained on Anti-UAV + DroneVsBird ─┐
camera ─┤                                                  ├─→ alert
        └─ LocateAnything-3B as ROI verifier               ┘
```

Where:

* The YOLO (or similar small-object detector) runs at full camera
  fps on the raw frame and emits candidate ROIs.
* LocateAnything-3B receives the cropped ROI and confirms /
  classifies / labels it via a referring prompt.

This project provides the *second* of those two components, set up
correctly. The first is out of scope here — but
`docs/CLIENT_PROTOCOL.md` shows exactly how a YOLO-based primary
detector should pipe its ROIs into this server.

For demo purposes (household objects) the model is excellent on its
own — those categories are smack in the center of its training
distribution.
