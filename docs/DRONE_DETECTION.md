# Drone detection — capabilities, limits, and honest assessment

This document is here because the project was set up with the
intent of building **ground-based FPV-drone early-warning** on top
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
  camera in daylight. Use `Point to: drone in the sky.` in `slow`
  mode.

* **Verifying / classifying ROIs already cropped by another
  detector.** When the drone occupies a significant fraction of
  the crop, LocateAnything's open-vocab strength kicks in.
  Prompt: `Locate a single instance that matches the following
  description: a quadcopter drone.` on the ROI.

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
  and prompt `Locate all the instances that match the following
  description: a small drone in the sky.`
* Parse each Result's `detections[].bbox_px` and translate back into
  source-image coords using the tile's offset.
* Apply NMS across all tiles to merge duplicates from the overlap
  bands.
* **Temporally smooth** over 3-5 frames before declaring a track —
  the model has no temporal model itself.

This multiplies inference cost by 12 per frame. At an indicative
single-frame `slow`-mode throughput of ~0.5 FPS, that's ~0.04 FPS
end-to-end. The geometry helpers in
[`worker/tiling.py`](../worker/tiling.py) (grid generation, IoU, NMS)
are ready for adoption into such a client.

## Throughput on RTX 5090 (estimated)

Estimated from the boot-time calibration on a clean snapshot:

* Single-frame, `hybrid` mode, 1080p image, 5-category prompt:
  ~0.5–1.0 s per frame ⇒ ~1–2 fps.
* Single-frame, `slow` mode, same input:
  ~1.5–3.0 s per frame ⇒ ~0.3–0.6 fps.
* 4 × 3 tiled, `slow` mode, 4K source:
  ~20–40 s per frame ⇒ ~0.03–0.05 fps.

The actual measured values are printed at container boot and
returned by `GET /v1/capabilities` under `.calibration`.

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
