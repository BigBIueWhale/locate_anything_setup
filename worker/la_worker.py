#!/usr/bin/env python3
"""
LocateAnything-3B inference sidecar.

Listens on a Unix domain socket (LA_IPC_SOCKET). Each connection from the
Rust frontend follows the protocol described in rust_server/src/protocol.rs:

  Each request = TWO consecutive length-prefixed frames sent by the client:
    1) JSON header
    2) JPEG bytes (may be zero-length for control messages)

  Each response = ONE length-prefixed frame:
    JSON response body

Length prefix: 4-byte big-endian unsigned int.

The model is loaded ONCE at startup. Inference is serialized through an
asyncio.Lock (PyTorch on CUDA is GIL-bound and can run at most one .generate
at a time). Frame ordering is FIFO across all connections — earlier frames
finish before later ones.

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

    # Bound the total number of concurrent UDS connections. Each one
    # spawns its own `handle_connection` Task; uncapped, anything in
    # the container that can connect to /tmp/la.sock could fan out
    # arbitrarily. The Rust frontend in steady state opens at most
    # a few connections (one per WS client, plus control queries),
    # so 16 is generous.
    MAX_CONCURRENT_CONNS = 16

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
        self._conn_sem = asyncio.Semaphore(self.MAX_CONCURRENT_CONNS)

    def capabilities(self) -> dict:
        # Single canonical capabilities response. The Rust frontend
        # forwards this verbatim to the client on Hello.
        gen = self.engine.gen_cfg
        return {
            "protocol_version": 1,
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
                "generation_mode":    gen.generation_mode,
                "n_future_tokens":    gen.n_future_tokens,
            },
            "supported_generation_modes": ["fast", "hybrid", "slow"],
            "supported_image_encodings":  ["jpeg"],
            "supported_color_spaces":     ["RGB"],
            "calibration": self.calibration.to_json(),
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
        if self._conn_sem.locked():
            log.warning(
                "concurrent connection cap %d reached; refusing peer %s",
                self.MAX_CONCURRENT_CONNS, peer,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return
        async with self._conn_sem:
            await self._handle_connection_inner(reader, writer, peer)

    async def _handle_connection_inner(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        peer,
    ) -> None:
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
                        "worker_protocol",
                        f"header bytes are not valid UTF-8: {e!s}. "
                        "First 32 bytes (hex): "
                        + header_bytes[:32].hex(),
                        code=400, retriable=False,
                    ))
                    continue
                except json.JSONDecodeError as e:
                    await write_frame(writer, _err_json(
                        "worker_protocol",
                        f"header JSON parse failed at line {e.lineno}, "
                        f"column {e.colno} (offset {e.pos}): {e.msg}",
                        code=400, retriable=False,
                    ))
                    continue
                if not isinstance(header, dict):
                    await write_frame(writer, _err_json(
                        "worker_protocol",
                        f"header JSON must be an object; got "
                        f"{type(header).__name__}",
                        code=400, retriable=False,
                    ))
                    continue

                # ---- Route ----
                kind = header.get("kind") or header.get("type")
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
                            "worker_protocol",
                            f"unknown header.kind: {kind!r}",
                            code=400, retriable=False,
                        ))
                except ValueError as e:
                    await write_frame(writer, _err_json(
                        "invalid_image", str(e), code=400, retriable=False
                    ))
                except RuntimeError as e:
                    log.exception("inference RuntimeError")
                    await write_frame(writer, _err_json(
                        "worker_error", str(e), code=500, retriable=True
                    ))
                except Exception as e:
                    log.exception("inference unexpected error")
                    await write_frame(writer, _err_json(
                        "worker_error", repr(e), code=500, retriable=True
                    ))
        finally:
            log.info("connection closed for %s", peer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _infer(self, header: dict, jpeg: bytes) -> dict:
        # The Rust frontend has already enforced the header schema and
        # the JPEG header validity. We re-validate here as a defense-in-
        # depth measure: this worker is also reachable by anything inside
        # the container that can touch /tmp/la.sock. There is no implicit
        # trust of the upstream.

        for key in ("prompt", "generation_mode", "frame_id", "session_id"):
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
                "docs/MODEL_CAPABILITIES.md for the canonical prompt forms)"
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

        # PyTorch / CUDA cannot run two .generate() concurrently. Serialize.
        async with self.lock:
            # The blocking inference runs in a thread so the asyncio loop
            # remains responsive for other connections (queueing). Each
            # call holds the lock for its full duration so frames are
            # processed strictly FIFO across all connections.
            #
            # IMPORTANT: asyncio.to_thread futures are NOT cancellable
            # once the thread has started running (concurrent.futures
            # contract). If our handle_connection Task is cancelled
            # while this await is pending, naive code would release
            # the lock immediately while the orphaned thread keeps
            # using the GPU — a *second* _infer would then collide on
            # the same model, scribbling the KV cache or hitting CUDA
            # OOM. To preserve the invariant "lock held ⇔ GPU busy",
            # we shield the inference Task and, on cancellation, DRAIN
            # the thread synchronously before letting the lock release.
            # The drain itself uses shield + a poll loop so a SECOND
            # cancellation (e.g. asyncio.run finalizing during
            # shutdown) doesn't interrupt the drain.
            t0 = time.perf_counter()
            inference_task = asyncio.ensure_future(
                asyncio.to_thread(self.engine.run, jpeg, prompt, mode),
            )
            try:
                result = await asyncio.shield(inference_task)
            except asyncio.CancelledError:
                while not inference_task.done():
                    try:
                        await asyncio.shield(inference_task)
                    except BaseException:
                        # absorbs both CancelledError and any exception
                        # the inference itself raised — we only care
                        # about reaching done() before re-raising
                        pass
                raise
            total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": True,
            "frame_id":      header["frame_id"],
            "session_id":    header["session_id"],
            "raw_text":      result.raw_answer,
            "detections":    result.detections,
            "points":        result.points,
            "abstained":     result.abstained,
            "image_size":    {"w": result.image_size[0], "h": result.image_size[1]},
            "resize_plan":   result.resize_plan,
            "generation_mode_used": mode,
            "latency_ms":    round(result.latency_ms, 1),
            "total_ms":      round(total_ms, 1),
        }


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _err_json(error_type: str, message: str, *, code: int, retriable: bool) -> bytes:
    return json.dumps({
        "ok":          False,
        "code":        code,
        "error_type":  error_type,
        "message":     message,
        "retriable":   retriable,
    }).encode("utf-8")


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
        default=os.environ.get("LA_CALIBRATION_IMAGE",
                               "/opt/locate_anything/test_data/calibration.jpg"),
    )
    p.add_argument(
        "--calibration-prompt",
        default=prompts.detect_categories(["person", "laptop", "bottle", "cup", "book"]),
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
