"""
Boot-time throughput calibration.

Runs the loaded inference engine N times on a fixed test image with a
fixed prompt — measures median + p95 per-frame latency, then publishes
the median sustainable FPS. The published FPS is purely advisory; the
server still processes every frame the client sends (no time-based drop).

Calibration uses the canonical trained generation kwargs in `hybrid` mode.
If the user has built the image with a different `LA_GEN_MODE`, we still
calibrate `hybrid` so the numbers are comparable across builds.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import statistics
import time

from .inference import LocateAnythingInference
from .parsing import has_abstention
from . import prompts


@dataclass(frozen=True)
class CalibrationResult:
    n_runs: int
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    median_fps: float
    test_image_path: str
    test_prompt: str

    def to_json(self) -> dict:
        return {
            "n_runs":              self.n_runs,
            "median_latency_ms":   round(self.median_latency_ms, 1),
            "p95_latency_ms":      round(self.p95_latency_ms, 1),
            "min_latency_ms":      round(self.min_latency_ms, 1),
            "max_latency_ms":      round(self.max_latency_ms, 1),
            "median_fps":          round(self.median_fps, 3),
            "test_image":          self.test_image_path,
            "test_prompt":         self.test_prompt,
        }


def calibrate(
    engine: LocateAnythingInference,
    test_image_path: str,
    test_prompt: str,
    n_runs: int = 6,
) -> CalibrationResult:
    """Run N inferences on a known image, return latency stats.

    Crashes if the test image is missing or unreadable — we do NOT fall
    back to a zeroed result, because callers use `calibration.median_fps`
    as ground truth for the GPU's sustainable rate. A bogus zero would
    silently mislead clients.
    """
    p = Path(test_image_path)
    if not p.is_file():
        raise RuntimeError(
            f"Calibration image not found at {p}. The container's "
            "self-test refuses to start without it — see "
            "scripts/01_download_weights.sh for how the image is "
            "generated."
        )
    try:
        jpeg = p.read_bytes()
    except OSError as e:
        raise RuntimeError(
            f"Calibration image at {p} is unreadable: {e!r}"
        ) from e
    if not jpeg:
        raise RuntimeError(
            f"Calibration image at {p} is zero bytes — re-generate it via "
            "scripts/01_download_weights.sh."
        )
    if n_runs < 1:
        raise ValueError(f"n_runs={n_runs} must be ≥ 1")

    # Classify the calibration prompt once; engine.run() requires the
    # wire name as a positional argument. The Rust validator does the
    # same classification at request time; prompts.classify_prompt is
    # the in-process Python mirror (raises if the prompt isn't one of
    # the seven canonical templates, caught early at boot rather than
    # per-request).
    test_prompt_task = prompts.classify_prompt(test_prompt)

    # Warm-up: first inference triggers extra JIT / cuDNN autotune overhead.
    print(f"[calibrate] warmup run on {p}", flush=True)
    warm = engine.run(jpeg, test_prompt, generation_mode="hybrid",
                      prompt_task=test_prompt_task)
    # The default calibration target is a real drone JPEG + `point_to`
    # drone prompt (see `worker/la_worker.py` argparse defaults) so the
    # published `median_fps` is workload-representative. Whether the
    # model actually detects the drone in the warm-up frame is
    # incidental — what we DO require is evidence that the model emitted
    # SOME structured output that the parser was able to consume. One of:
    # (a) at least one parsed box, (b) at least one parsed point,
    # (c) the trained explicit abstention literal `<box>None</box>`
    #     present in raw_text. Anything else means raw_text was
    #     unparseable, which would be a model-or-parser bug we want to
    #     catch at boot — not in production.
    #
    # We deliberately invoke `has_abstention(warm.raw_answer)` here rather
    # than reading `warm.abstained`. The aggregate `warm.abstained` is
    # `not (detections or points)` and would always be True when both
    # lists are empty — making the disjunction tautological and silently
    # masking the gibberish-output failure mode this assertion exists to
    # catch. The substring scan via `has_abstention` is the parser-internal
    # probe for "did the model
    # emit the trained literal at all", which is what we actually need
    # for the parser-drift self-test.
    if not (warm.detections or warm.points or has_abstention(warm.raw_answer)):
        raise RuntimeError(
            "calibration warm-up yielded no recognizable output: no parsed "
            "boxes, no parsed points, and no `<box>None</box>` abstention "
            "literal in raw_text. raw_text first 200 chars: "
            f"{warm.raw_answer[:200]!r}. Either the model is mis-loaded, "
            "the prompt path is broken, or the parser regex needs an "
            "update for new output forms."
        )

    # Note: engine.run() already cuda.synchronize()'s around its inner
    # .generate(). The Python-side measurement here adds the JPEG decode
    # and preprocessing overhead; cuda sync inside engine.run keeps the
    # latency honest.
    latencies_ms: List[float] = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = engine.run(jpeg, test_prompt, generation_mode="hybrid",
                            prompt_task=test_prompt_task)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        print(f"[calibrate] run {i+1}/{n_runs}: {latencies_ms[-1]:.1f} ms, "
              f"{len(result.detections)} boxes, {len(result.points)} points, "
              f"abstain={result.abstained}",
              flush=True)

    latencies_ms.sort()
    median = statistics.median(latencies_ms)
    # p95 by index for small N — for n=6 this is index 5 (last value).
    p95_idx = max(0, int(0.95 * (len(latencies_ms) - 1) + 0.5))
    p95 = latencies_ms[p95_idx]
    median_fps = 1000.0 / median if median > 0 else 0.0

    return CalibrationResult(
        n_runs=n_runs,
        median_latency_ms=median,
        p95_latency_ms=p95,
        min_latency_ms=latencies_ms[0],
        max_latency_ms=latencies_ms[-1],
        median_fps=median_fps,
        test_image_path=str(p),
        test_prompt=test_prompt,
    )
