#!/usr/bin/env python3
"""
LocateAnything-3B inference sidecar.

Listens on a Unix domain socket (LA_IPC_SOCKET). Each Rust→Python request
is TWO consecutive length-prefixed frames sent by the Rust side:
    1) JSON header
    2) JPEG bytes (zero-length for control queries)
The worker responds with ONE length-prefixed frame: a JSON body. The
shape depends on the header.kind:
    "frame"        → either a successful inference body or {code, message}
                      on failure. The Rust frontend adds type+frame_id.
    "capabilities" → /v1/capabilities payload (no `type` field).
    "info"         → /v1/info payload.

Length prefix: 4-byte big-endian unsigned int.

The model is loaded once at startup. Inference is serialized through an
asyncio.Lock — PyTorch on CUDA can only run one .generate() at a time.
Multiple Rust→Python connections fan out concurrently; the lock turns
that into a fair FIFO queue.

This file does NOT contain a fallback for any error. If model load fails,
GPU is missing, or weights are corrupt, we exit non-zero.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import torch  # only used at module level for torch.OutOfMemoryError catch
# We import these eagerly so torch/transformers initialization happens
# during the visible startup phase, not on the first request.
from . import validate_startup
from .calibration import calibrate, CalibrationResult
from .inference import LocateAnythingInference
from .pixel_token_math import summarize as pixel_token_summary
from . import prompts


LOG_FMT = "%(asctime)s %(levelname)s [worker] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)
log = logging.getLogger("la_worker")


# -------------------------------------------------------------------------
# Hard-recovery contract (worker self-exit → container restart)
# -------------------------------------------------------------------------
#
# Two server-side failure modes fail loud rather than silently degrade:
# per-frame inference timeout and CUDA OutOfMemoryError. Both call
# os._exit(WORKER_RESTART_EXIT_CODE) so entrypoint.sh's `wait -n`
# propagates the exit through to Docker's `--restart unless-stopped`.
# Typical visible gap: ~10–15 s for model reload + boot drift check +
# drone calibration. Operator-facing details (per-failure client
# signature, persistent-failure restart loop):
# docs/OPERATIONS.md §Worker self-exit.
#
# WHY hard-exit instead of in-process recovery — for each:
#
#   * Inference timeout: CPython provides no mechanism to cancel a
#     thread blocked inside a non-Python C/CUDA frame holding the GIL.
#     `asyncio.wait_for` releases the asyncio.Lock cleanly on expiry,
#     but the underlying thread keeps running and keeps holding the
#     GPU. Process exit is the only mechanism that kills the orphan.
#
#   * CUDA OOM: PyTorch's caching allocator IS empirically stable
#     across repeated OOMs on PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#     :True, but empirical stability is not provable correctness. The
#     principled choice is to exit so the next frame runs against a
#     fresh CUDA context.

# Per-frame inference timeout in seconds; env-configurable. Default 600 s
# is the theoretical maximum legitimate inference latency for this model
# at the trained sampling parameters: max_new_tokens (8192) divided by
# the observed minimum slow-mode tokens-per-second (~30) plus prefill
# (~0.5 s) plus 2× safety, ≈ 600 s. The single longest-output legitimate
# input characterised on this server (a synthetic dense-text grid that
# saturates max_new_tokens in slow mode) ran for ~273 s; the 600 s bound
# never falsely kills a legitimate workload. Wedge-detection latency is
# bounded by the same value — accepted as the principled trade.
LA_INFERENCE_TIMEOUT_S = float(os.environ.get("LA_INFERENCE_TIMEOUT_S", "600"))

# EX_TEMPFAIL from sysexits.h — "service unavailable, retry recommended".
# Chosen over 137 (SIGKILL convention) and 99 (project-internal) for the
# clean POSIX-sysexits semantic; an operator grepping logs can find it
# documented in /usr/include/sysexits.h:78.
WORKER_RESTART_EXIT_CODE = 75


def _hard_exit_for_restart(code: int = WORKER_RESTART_EXIT_CODE) -> None:
    """Flush log handlers, stdout, stderr — then call os._exit(code).

    We use os._exit (not sys.exit) so atexit handlers are skipped:
    atexit runs Python finalization which can itself hang on a wedged
    CUDA call, defeating the whole point of the timeout. Log flushing
    matters because os._exit also skips stdio flushing, and we want
    the diagnostic line to actually reach the container log before the
    process dies."""
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    try:
        sys.stderr.flush()
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)


# -------------------------------------------------------------------------
# Framing
# -------------------------------------------------------------------------

LEN_PREFIX = 4
MAX_FRAME_BYTES = 16 * 1024 * 1024


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one length-prefixed frame. Raises ConnectionError on EOF mid-frame."""
    header = await reader.readexactly(LEN_PREFIX)
    (size,) = struct.unpack(">I", header)
    if size > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {size} > {MAX_FRAME_BYTES}")
    if size == 0:
        return b""
    return await reader.readexactly(size)


async def write_frame(writer: asyncio.StreamWriter, data: bytes) -> None:
    """Write one length-prefixed frame. Short-circuits with ConnectionError
    if the transport is already closing — saves the broad-except cascade
    in the handler when a client disconnected mid-inference."""
    if writer.is_closing():
        raise ConnectionError("writer is closing")
    if len(data) > MAX_FRAME_BYTES:
        raise ValueError(f"outbound frame too large: {len(data)}")
    writer.write(struct.pack(">I", len(data)))
    writer.write(data)
    await writer.drain()


# -------------------------------------------------------------------------
# Application
# -------------------------------------------------------------------------

class WorkerApp:
    """The single inference engine + a lock that serializes GPU access."""

    def __init__(
        self,
        engine: LocateAnythingInference,
        model_manifest_sha256: Optional[str],
        calibration: CalibrationResult,
    ):
        self.engine = engine
        self.lock = asyncio.Lock()
        self.model_manifest_sha256 = model_manifest_sha256
        self.calibration = calibration

    def capabilities(self) -> dict:
        # Served verbatim by Rust on GET /v1/capabilities.
        gen = self.engine.gen_cfg
        return {
            "model":                 "nvidia/LocateAnything-3B",
            "model_dir":             self.engine.model_dir,
            # Manifest hash = SHA-256 over a sorted list of (filename, size)
            # in the bind-mounted model directory. It detects renamed,
            # truncated, or added files between boots — it is NOT a content
            # hash of the safetensors. Named *_manifest_sha256 so no one
            # confuses it with a model content hash.
            "model_manifest_sha256": self.model_manifest_sha256,
            "dtype":            "bfloat16",
            "attn_impl":        self.engine.attn_impl,
            "max_image_dim":    2240,         # see docs/PIXEL_TO_TOKEN_MATH.md
            "in_token_limit":   25600,
            "max_llm_tokens_per_image": 6400,
            "patch_px":         14,
            "llm_token_px":     28,
            "max_prompt_tokens": 16384,       # tokenizer.model_max_length
            "trained_generation_params": {
                "do_sample":          gen.do_sample,
                "temperature":        gen.temperature,
                "top_p":              gen.top_p,
                "repetition_penalty": gen.repetition_penalty,
                "max_new_tokens":     gen.max_new_tokens,
                "n_future_tokens":    gen.n_future_tokens,
            },
            "supported_generation_modes": ["fast", "hybrid", "slow"],
            "calibration": self.calibration.to_json(),
            # The single-source-of-truth file for the seven canonical
            # LocateAnything-3B prompt templates. Every prompt the server
            # accepts MUST conform to one of those templates — strictly
            # enforced at the WebSocket edge by the Rust validator
            # (rust_server/src/prompt_validator.rs), with the same URL
            # included in every rejection diagnostic so the client knows
            # where to look. Per the project policy "only use the model
            # how it was trained".
            "prompt_templates_reference_url": prompts.CANONICAL_REFERENCE_URL,
            "preset_prompts": {
                "drone_ranked": prompts.DRONE_PROMPTS_RANKED,
                "household":    prompts.HOUSEHOLD_PROMPTS,
            },
        }

    def info(self) -> dict:
        free_b, total_b = _gpu_mem_free_total_bytes()
        return {
            "ok": True,
            "model_loaded": True,
            "torch_arches":       _torch_arches(),
            "gpu_name":           _gpu_name(),
            "gpu_total_mem_gib":  round(total_b / (1024 ** 3), 2),
            "gpu_free_mem_gib":   round(free_b  / (1024 ** 3), 2),
            "gpu_used_mem_gib":   round((total_b - free_b) / (1024 ** 3), 2),
            "pixel_token_examples": {
                "1080p":       pixel_token_summary(1920, 1080),
                "1440p":       pixel_token_summary(2560, 1440),
                "4K":          pixel_token_summary(3840, 2160),
                "square_2240": pixel_token_summary(2240, 2240),
            },
        }

    async def handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "<uds>"
        log.info("connection open from %s", peer)
        try:
            while True:
                # ---- Read request: two frames (header JSON, payload) ----
                # Once we've started reading either frame, ANY exception
                # leaves the framed stream in an indeterminate position
                # and the next iteration's read_frame would interpret
                # arbitrary bytes as a length prefix. We treat any read-
                # path exception as connection-fatal: close the
                # connection and let the peer reconnect with a fresh
                # framed stream. This preserves the framing invariant
                # at the cost of one reconnect per partial read — which
                # is the right trade for not silently poisoning future
                # requests on the same connection.
                try:
                    header_bytes = await read_frame(reader)
                except asyncio.IncompleteReadError:
                    log.info("peer closed cleanly between requests (%s)", peer)
                    break
                try:
                    payload_bytes = await read_frame(reader)
                except (asyncio.IncompleteReadError, ValueError, ConnectionError,
                        struct.error) as e:
                    log.warning(
                        "UDS payload read failed mid-request (%s); the framed "
                        "stream is now desynced — closing connection so the "
                        "client can reconnect. cause=%r",
                        peer, e,
                    )
                    break

                # ---- Parse header ----
                try:
                    header = json.loads(header_bytes.decode("utf-8"))
                except UnicodeDecodeError as e:
                    await write_frame(writer, _err_json(
                        400,
                        f"header bytes are not valid UTF-8: {e!s}. "
                        "First 32 bytes (hex): "
                        + header_bytes[:32].hex(),
                    ))
                    continue
                except json.JSONDecodeError as e:
                    await write_frame(writer, _err_json(
                        400,
                        f"header JSON parse failed at line {e.lineno}, "
                        f"column {e.colno} (offset {e.pos}): {e.msg}",
                    ))
                    continue
                if not isinstance(header, dict):
                    await write_frame(writer, _err_json(
                        400,
                        f"header JSON must be an object; got "
                        f"{type(header).__name__}",
                    ))
                    continue

                # ---- Route ----
                # The Rust IPC layer always stamps `kind`; absent or wrong-
                # cased = upstream contract violation, not a client mistake.
                kind = header.get("kind")
                try:
                    if kind == "capabilities":
                        await write_frame(writer, json.dumps(self.capabilities()).encode())
                    elif kind == "info":
                        await write_frame(writer, json.dumps(self.info()).encode())
                    elif kind == "frame":
                        resp = await self._infer(header, payload_bytes)
                        await write_frame(writer, json.dumps(resp).encode())
                    else:
                        await write_frame(writer, _err_json(
                            400, f"unknown header.kind: {kind!r}",
                        ))
                except torch.OutOfMemoryError as e:
                    # CUDA OOM. Must catch BEFORE RuntimeError (OOM is a
                    # RuntimeError subclass). Write a typed per-frame
                    # error so the client knows specifically why, then
                    # hard-exit — see "Hard-recovery contract" at top
                    # of file.
                    log.error(
                        "CUDA OutOfMemoryError during inference: %s — exiting %d for container restart",
                        e, WORKER_RESTART_EXIT_CODE,
                    )
                    try:
                        await write_frame(writer, _err_json(
                            500,
                            f"CUDA out of memory: {e}. The worker is "
                            "restarting; retry the request after ~10-15s.",
                        ))
                    except Exception as write_err:
                        log.warning(
                            "failed to write OOM error to client before exit: %r",
                            write_err,
                        )
                    _hard_exit_for_restart()
                except asyncio.TimeoutError:
                    # Per-frame inference timeout. asyncio.Lock is already
                    # released by the `async with` unwind, but the orphan
                    # thread keeps running — only process exit kills it.
                    # See "Hard-recovery contract" at top of file.
                    log.error(
                        "inference exceeded LA_INFERENCE_TIMEOUT_S=%.1fs — exiting %d for container restart",
                        LA_INFERENCE_TIMEOUT_S, WORKER_RESTART_EXIT_CODE,
                    )
                    try:
                        await write_frame(writer, _err_json(
                            504,
                            f"inference timeout: exceeded "
                            f"LA_INFERENCE_TIMEOUT_S={LA_INFERENCE_TIMEOUT_S}s. "
                            "The worker is restarting; retry the request "
                            "after ~10-15s.",
                        ))
                    except Exception as write_err:
                        log.warning(
                            "failed to write timeout error to client before exit: %r",
                            write_err,
                        )
                    _hard_exit_for_restart()
                except ValueError as e:
                    await write_frame(writer, _err_json(400, str(e)))
                except RuntimeError as e:
                    log.exception("inference RuntimeError")
                    await write_frame(writer, _err_json(500, str(e)))
                except Exception as e:
                    log.exception("inference unexpected error")
                    await write_frame(writer, _err_json(500, repr(e)))
        finally:
            log.info("connection closed for %s", peer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _infer(self, header: dict, jpeg: bytes) -> dict:
        # The Rust frontend has already enforced the header schema and JPEG
        # SOI marker. We re-validate here as defense in depth: this worker is
        # also reachable by anything inside the container that can touch
        # /tmp/la.sock.
        for key in ("prompt", "generation_mode", "frame_id", "prompt_task"):
            if key not in header:
                raise ValueError(
                    f"header.{key} missing. The Rust frontend should have "
                    "rejected this Frame upstream — receiving it here means "
                    "the upstream contract was violated."
                )
        prompt = header["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "header.prompt must be a non-empty string (see "
                "worker/prompts.py for the seven canonical prompt forms — "
                "that file is the single source of truth)"
            )
        if len(prompt) > 16384:
            raise ValueError(
                f"header.prompt length {len(prompt)} chars > 16384 "
                "(the tokenizer's model_max_length cap)"
            )
        mode = header["generation_mode"]
        if mode not in ("fast", "hybrid", "slow"):
            raise ValueError(
                f"header.generation_mode={mode!r} is not one of "
                "'fast'|'hybrid'|'slow'. There is no default — every "
                "request must declare a mode explicitly."
            )
        if not isinstance(jpeg, (bytes, bytearray)) or not jpeg:
            raise ValueError(
                "JPEG payload is empty. The Rust frontend should have "
                "rejected this Frame; receiving it here means the "
                "upstream contract was violated."
            )
        # Defense-in-depth JPEG SOI sniff. Rust already enforces this; we
        # double-check here because /tmp/la.sock is also reachable by
        # anything inside the container.
        if jpeg[:3] != b"\xff\xd8\xff":
            raise ValueError(
                "JPEG payload missing SOI marker FF D8 FF — Rust frontend "
                "should have rejected this Frame; defense-in-depth check "
                "tripped here because /tmp/la.sock is internally reachable."
            )

        # PyTorch/CUDA cannot run two .generate() concurrently. Serialize.
        # Once a frame enters _infer, the inference runs to completion (up
        # to LA_INFERENCE_TIMEOUT_S seconds) regardless of client liveness.
        # If the WS has closed by the time we go to write the response, the
        # write fails into the closed UDS and the next frame proceeds.
        #
        # The asyncio.wait_for raises asyncio.TimeoutError on expiry; we do
        # NOT catch it here — it propagates up to handle_connection's
        # exception block which writes a per-frame error to the client AND
        # hard-exits the worker (see "Hard-recovery contract" at the top of
        # this file). The asyncio.Lock IS released cleanly on the exception
        # (the `async with` unwind runs), but the underlying thread keeps
        # executing engine.run because Python provides no mechanism to
        # cancel a thread inside a CUDA C call — the orphan dies with the
        # process when _hard_exit_for_restart calls os._exit.
        prompt_task = header["prompt_task"]
        async with self.lock:
            t0 = time.perf_counter()
            result = await asyncio.wait_for(
                asyncio.to_thread(self.engine.run, jpeg, prompt, mode, prompt_task),
                timeout=LA_INFERENCE_TIMEOUT_S,
            )
            total_ms = (time.perf_counter() - t0) * 1000.0
        # `ok` is an IPC-only discriminator the Rust frontend strips before
        # stamping type+frame_id; it never appears in the client-facing body.
        return {
            "ok":            True,
            "raw_text":      result.raw_answer,
            "detections":    result.detections,
            "points":        result.points,
            "abstained":     result.abstained,
            # Count of geometries the model emitted in the WRONG shape for the
            # task (filtered out of detections/points). Normally 0; non-zero is
            # a loud model task->shape deviation signal — NOT abstention.
            "off_shape_count": result.off_shape_count,
            # Wire name of the canonical template the prompt was
            # classified as by the Rust validator. Echoed to the client
            # so they can branch on `prompt_task == "point"` (→ read
            # `points[]`) vs any other value (→ read `detections[]`)
            # without re-classifying the prompt themselves.
            "prompt_task":   result.prompt_task,
            # True iff `raw_text` does NOT end with the model's <|im_end|>
            # end-of-turn marker, meaning the custom .generate() loop
            # terminated because max_new_tokens was reached, NOT because
            # the model cleanly finished. Per
            # models/LocateAnything-3B/modeling_locateanything.py:464,500-501
            # the loop exits ONLY on <|im_end|> emission OR budget
            # exhaustion, so this is a total signal. Naive clients can
            # branch on this typed boolean instead of substring-checking
            # raw_text themselves.
            "model_output_truncated": result.model_output_truncated,
            "image_size":    [result.image_size[0], result.image_size[1]],
            "resize_plan":   result.resize_plan,
            "generation_mode_used": mode,
            "latency_ms":    round(result.latency_ms, 1),
            "total_ms":      round(total_ms, 1),
        }


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _err_json(code: int, message: str) -> bytes:
    """Worker-side error body. `ok:false` is the IPC discriminator the Rust
    frontend uses to route to type:"error"; the client sees
    `{type, frame_id, code, message}` after Rust strips `ok`."""
    return json.dumps({"ok": False, "code": code, "message": message}).encode("utf-8")


def _torch_arches() -> list:
    import torch
    return list(torch.cuda.get_arch_list())


def _gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0)


def _gpu_mem_free_total_bytes() -> tuple:
    """Live free/total VRAM in bytes from the CUDA driver. Exposes the value
    operators actually care about during a degraded incident."""
    import torch
    free_b, total_b = torch.cuda.mem_get_info(0)
    return int(free_b), int(total_b)


def _compute_weight_manifest_sha256(model_dir: Path) -> Optional[str]:
    """SHA-256 over the concatenated sorted-list of (filename, size) — NOT
    a content hash. Real content hashing of 7 GB at boot is wasteful.
    The point is to detect 'someone swapped the weights' between boots.
    Returns None if the directory is empty."""
    import hashlib
    h = hashlib.sha256()
    items = []
    for f in sorted(model_dir.glob("*")):
        if f.is_file():
            items.append((f.name, f.stat().st_size))
    if not items:
        return None
    for name, sz in items:
        h.update(f"{name}:{sz}\n".encode("utf-8"))
    return h.hexdigest()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

async def main_async(args) -> int:
    log.info("validating environment…")
    validate_startup.run_all(args.model_dir)

    log.info("loading model from %s…", args.model_dir)
    engine = LocateAnythingInference(args.model_dir, device="cuda")
    log.info("model loaded; running boot calibration…")

    calib = calibrate(engine, args.calibration_image, args.calibration_prompt, n_runs=args.calibration_runs)
    log.info(
        "calibration: median %.1f ms (~%.2f FPS), p95 %.1f ms",
        calib.median_latency_ms, calib.median_fps, calib.p95_latency_ms,
    )

    weight_manifest_sha = _compute_weight_manifest_sha256(Path(args.model_dir))
    app = WorkerApp(engine, weight_manifest_sha, calib)

    # Remove stale socket if present.
    sock_path = Path(args.socket)
    if sock_path.exists():
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass

    # Tighten umask BEFORE the bind so the socket is created 0o660
    # atomically. Without this, there is a sub-millisecond window where
    # the socket is world-rw between socket(2) and chmod(2). The container
    # has no other user (only `la` exists), so the practical impact is
    # nil, but defense-in-depth costs one line.
    old_umask = os.umask(0o117)
    try:
        server = await asyncio.start_unix_server(
            app.handle_connection, path=str(sock_path),
        )
    finally:
        os.umask(old_umask)
    # Belt for the umask suspenders — if the bind succeeded but the perm
    # bits are wrong for any reason, force them now.
    os.chmod(str(sock_path), 0o660)
    log.info("listening on %s", sock_path)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # SIGHUP added — outside Docker (systemd, runit), SIGHUP would
    # otherwise terminate the process abruptly without running the
    # finally block (no clean socket unlink, no inference drain).
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(s, stop.set)

    async with server:
        try:
            await stop.wait()
        finally:
            log.info("shutting down — closing UDS listener")
            server.close()
            await server.wait_closed()
            # Drain any in-flight inference before the loop tears down.
            # _infer holds app.lock for the duration of an inference; if
            # we acquire it here, no inference is in flight. 60s is a
            # very generous upper bound — even the slowest hybrid-mode
            # run completes well under that.
            log.info("draining in-flight inference (up to 60s)")
            try:
                await asyncio.wait_for(app.lock.acquire(), timeout=60.0)
                app.lock.release()
                log.info("inference drained cleanly")
            except asyncio.TimeoutError:
                log.warning("inference drain timeout — process will exit with active GPU work")
            # Explicit ThreadPoolExecutor shutdown so any to_thread
            # workers exit before the interpreter tears down, keeping
            # exit logs clean.
            try:
                await loop.shutdown_default_executor()
            except Exception as e:
                log.warning("default executor shutdown error: %r", e)
    try:
        sock_path.unlink()
    except FileNotFoundError:
        pass
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LocateAnything-3B Unix socket worker")
    p.add_argument(
        "--socket",
        default=os.environ.get("LA_IPC_SOCKET", "/tmp/la.sock"),
        help="Unix domain socket path to bind.",
    )
    p.add_argument(
        "--model-dir",
        default=os.environ.get("LA_MODEL_LOCAL_DIR", "/opt/locate_anything/model"),
        help="Directory containing the LocateAnything-3B HF snapshot.",
    )
    p.add_argument(
        "--calibration-image",
        # Default is a real drone JPEG so the published median_fps is
        # workload-representative for the documented primary use case
        # (per `docs/DRONE_DETECTION.md` Throughput table). The previous
        # default — a 1024×768 synthetic 4-polygon image — over-reported
        # by ~2.7× because it sits well below the typical patch budget;
        # the synthetic image is still generated by
        # `scripts/01_download_weights.sh` and used by the smoke tests
        # (where "we know what to detect" is the load-bearing property).
        default=os.environ.get("LA_CALIBRATION_IMAGE",
                               "/opt/locate_anything/test_data/drone_sirius.jpg"),
    )
    p.add_argument(
        "--calibration-prompt",
        # Paired with the drone calibration image. `point_to('drone in
        # the sky')` is the structurally cleanest drone prompt — it
        # produces few output tokens and triggers MTP's fast path
        # almost every block, so per-boot calibration latency
        # characterises the well-behaved fast-path workload.
        default=prompts.point_to("drone in the sky"),
    )
    p.add_argument(
        "--calibration-runs",
        type=int,
        default=int(os.environ.get("LA_CALIBRATION_RUNS", "6")),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
