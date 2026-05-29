#!/usr/bin/env python3
"""
Reference client for the LocateAnything-3B WebSocket protocol.

Demonstrates the correct way to:
  1. Fetch server limits once over HTTP (GET /v1/capabilities).
  2. Open WS and start sending Frames immediately — no handshake.
  3. Read frames from a source (file, V4L2, or RTSP).
  4. Send each frame with a correlated frame_id.
  5. Receive results / errors; correlate by frame_id.
  6. Respect TCP backpressure (sender awaits, no drop).
  7. Reconnect cleanly on close.

Run:
    pip install websockets opencv-python-headless httpx
    python reference_client.py --source path/to/video.mp4 \
        --prompt "Point to: drone in the sky."

Or against an RTSP stream:
    python reference_client.py --source rtsp://... --mode slow --prompt "..."

Reads frames in a background thread (cv2.VideoCapture is sync) and feeds
them into the WS via an asyncio.Queue. Backpressure happens at the WS
send: when the server stops draining, ws.send().await blocks, which
fills the queue, which makes the camera-reader thread block on
queue.put — natural end-to-end flow control.

This client does NOT do its own frame dropping. If the GPU can't keep
up, the client's capture cadence will be forced down by the server's
backpressure. If your use case requires "always process the most recent
frame", that decision goes in the CAPTURE LAYER (a deliberate
modulo-N decimation), not in the network layer — see
docs/CLIENT_PROTOCOL.md for the rationale.

The server is stateless across WebSockets: no per-session state lives
on the server. On reconnect the client just opens a new WS and resumes
sending Frames; frame_id namespacing is the client's prerogative.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import struct
import sys
import threading
import urllib.parse
import urllib.request
from queue import Queue

import cv2
import websockets

log = logging.getLogger("client")


def fetch_capabilities(ws_url: str, timeout: float = 10.0) -> dict:
    """Synchronous one-shot HTTP GET for /v1/capabilities. Derives the
    HTTP base from the WS URL (ws://host:port/path → http://host:port).
    Done before opening the WS so caps are available to the capture
    thread (which needs max_image_dim)."""
    parsed = urllib.parse.urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    url = f"{scheme}://{parsed.netloc}/v1/capabilities"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def encode_jpeg(frame_bgr, quality: int) -> bytes:
    """cv2's encoder is libjpeg-turbo. RGB→BGR conversion not needed when
    encoding (cv2 expects BGR which is its native order)."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def capture_thread(source: str, max_dim: int, q: Queue, stop_event: threading.Event) -> None:
    """Pull frames from cv2.VideoCapture and stuff them into the queue."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("could not open source %r", source)
        stop_event.set()
        return
    frame_idx = 0
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                # End of file or stream lost. Break — main loop decides reconnect.
                break
            h, w = frame.shape[:2]
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            # q.put blocks when full — that's the backpressure entry point.
            q.put((frame_idx, frame))
            frame_idx += 1
    finally:
        cap.release()
        q.put(None)  # sentinel


async def receiver(ws):
    """Print every server message keyed by frame_id where applicable."""
    async for raw in ws:
        try:
            obj = json.loads(raw)
        except Exception:
            log.warning("non-JSON message: %r", raw)
            continue
        t = obj.get("type")
        if t == "result":
            fid = obj.get("frame_id")
            dets = obj.get("detections", [])
            pts = obj.get("points", [])
            abst = obj.get("abstained", False)
            log.info(
                "frame %s: %d boxes / %d points / abstain=%s / %.1f ms",
                fid, len(dets), len(pts), abst, obj.get("latency_ms", 0.0),
            )
            for d in dets[:3]:
                log.info("  %s @ px=%s", d.get("label") or "<unlabeled>", d.get("bbox_px"))
        elif t == "error":
            log.error("error: code=%s msg=%s frame_id=%s",
                      obj.get("code"), obj.get("message"), obj.get("frame_id"))
        else:
            # The protocol emits only result / error on the WS. Anything
            # else is a server-side bug worth flagging loudly.
            log.warning("unrecognised server message type=%r: %r", t, obj)


async def sender(ws, q: Queue, prompt: str, generation_mode: str,
                 jpeg_quality: int, client_id: str,
                 stop_event: threading.Event):
    """Pull frames from the queue and send them over WS, with backpressure."""
    loop = asyncio.get_running_loop()
    while True:
        # asyncio.to_thread(q.get) keeps the asyncio loop responsive while
        # the underlying thread-safe Queue.get() blocks.
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            return
        idx, frame = item
        try:
            jpeg = await loop.run_in_executor(
                None, encode_jpeg, frame, jpeg_quality
            )
        except Exception as e:
            log.error("encode failed: %s", e)
            continue
        # client_id is namespaced into the frame_id locally; the wire
        # header carries only {frame_id, prompt, generation_mode, jpeg_len}.
        header = json.dumps({
            "frame_id":        f"{client_id}-{idx:08d}",
            "prompt":          prompt,
            "generation_mode": generation_mode,
            "jpeg_len":        len(jpeg),
        }).encode("utf-8")
        payload = struct.pack(">I", len(header)) + header + jpeg
        # This `await` is the network-side backpressure point — it will
        # block whenever the server stops draining.
        await ws.send(payload)


async def run_once(args, caps: dict):
    log.info("connecting to %s (client_id=%s)", args.url, args.client_id)
    async with websockets.connect(
        args.url,
        max_size=args.max_jpeg_bytes + 64 * 1024,
        open_timeout=15,
        ping_interval=20,
    ) as ws:
        q: Queue = Queue(maxsize=args.queue_max)
        stop_event = threading.Event()
        t = threading.Thread(
            target=capture_thread,
            args=(args.source, caps.get("max_image_dim", 2240), q, stop_event),
            daemon=True,
        )
        t.start()
        try:
            await asyncio.gather(
                sender(ws, q, args.prompt, args.mode, args.jpeg_quality,
                       args.client_id, stop_event),
                receiver(ws),
            )
        finally:
            stop_event.set()
            t.join(timeout=2.0)


async def main_async(args):
    """Reconnect with exponential backoff. A run of `max_consecutive_errors`
    failures aborts loudly — a persistent server-side problem must surface,
    not become silent log spam."""
    # One-shot capabilities fetch over HTTP — the server is stateless
    # across reconnects so caps don't need re-fetching on reconnect, but
    # if the server returns a hard error on capabilities the operator
    # needs to see that immediately.
    try:
        caps = await asyncio.to_thread(fetch_capabilities, args.url)
        log.info("server caps: model=%s, fps=%.2f, max_image_dim=%s",
                 caps.get("model"),
                 caps.get("calibration", {}).get("median_fps", 0.0),
                 caps.get("max_image_dim"))
    except Exception as e:
        log.error("fetch capabilities failed: %s: %s", type(e).__name__, e)
        sys.exit(2)
    base_delay = float(args.reconnect_delay)
    max_delay = max(base_delay, 60.0)
    consecutive_errors = 0
    while True:
        try:
            await run_once(args, caps)
            consecutive_errors = 0  # a clean run resets the counter
        except websockets.exceptions.ConnectionClosedOK as e:
            # Server initiated a normal close (code 1000) or going-away
            # (code 1001). Reconnect without counting toward the abort
            # threshold — a planned admin restart is not a fault.
            log.info("ws closed normally: code=%s reason=%r",
                     getattr(e, "code", None), getattr(e, "reason", None))
            consecutive_errors = 0
        except websockets.exceptions.ConnectionClosedError as e:
            # Abnormal close (1006 no Close frame; 1008 policy; 1011
            # server error). Counts toward abort.
            consecutive_errors += 1
            log.warning("ws closed abnormally (%d/%d): code=%s reason=%r",
                        consecutive_errors, args.max_consecutive_errors,
                        getattr(e, "code", None), getattr(e, "reason", None))
        except Exception as e:
            consecutive_errors += 1
            log.error("run_once failed (%d/%d): %s: %s",
                      consecutive_errors, args.max_consecutive_errors,
                      type(e).__name__, e)
        if args.no_reconnect:
            break
        if consecutive_errors >= args.max_consecutive_errors:
            log.error(
                "aborting after %d consecutive failures; the server-side "
                "issue is persistent and must be investigated before "
                "automatic recovery will succeed",
                consecutive_errors,
            )
            sys.exit(2)
        delay = min(max_delay, base_delay * (2 ** (consecutive_errors - 1)))
        log.info("reconnecting in %.1fs (exponential backoff)", delay)
        await asyncio.sleep(delay)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8765/v1/stream")
    p.add_argument("--source", default="0",
                   help="cv2 source: '0' webcam, path to file, rtsp://...")
    p.add_argument("--prompt", required=True,
                   help="Use one of the canonical prompt forms from /v1/capabilities.preset_prompts")
    p.add_argument("--mode", choices=("fast", "hybrid", "slow"), default="hybrid")
    p.add_argument("--client-id",  default="reference-client-01",
                   help="local label; namespaces frame_ids and appears in "
                        "this client's terminal output. NOT sent on the wire — "
                        "the server is stateless and doesn't know about clients.")
    p.add_argument("--max-jpeg-bytes", type=int, default=4 * 1024 * 1024)
    p.add_argument("--queue-max", type=int, default=4)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--reconnect-delay", type=int, default=3,
                   help="initial reconnect delay in seconds (exponential backoff multiplies)")
    p.add_argument("--max-consecutive-errors", type=int, default=10,
                   help="abort after this many back-to-back failures")
    p.add_argument("--no-reconnect", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    # cv2 accepts an integer as a webcam index when passed as int
    if args.source.isdigit():
        args.source = int(args.source)
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("ctrl-c, exiting")


if __name__ == "__main__":
    main()
