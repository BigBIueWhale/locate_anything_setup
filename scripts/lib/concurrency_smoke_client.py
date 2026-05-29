#!/usr/bin/env python3
"""
Concurrency smoke client for scripts/05_concurrency_smoke.sh.

Opens N concurrent WebSocket connections, each sending K frames sequentially
at fixed intervals (with a per-client start offset so frames interleave
across clients), records per-frame latency, and validates the FIFO
time-share invariant of the worker's single asyncio.Lock.

Failure modes this catches:
  - One client starves another (asyncio.Lock unfairness across UDS conns).
  - A response is delivered for the wrong frame_id (mux confusion).
  - A frame never receives a response (deadlock or worker hang).
  - The worker returns type:"error" for a normally-correct request under
    concurrent load (e.g. asyncio.shield drain bug, mid-inference KV
    cache collision from a missed lock).
  - Median latency varies wildly across clients (de-facto starvation).

Run via `docker exec` against the live container; NOT a reference for
production clients (see examples/reference_client.py for that). Wire format
follows scripts/lib/smoke_ws_client.py verbatim — these two clients MUST
stay in sync with the server-side protocol.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import statistics
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import websockets


@dataclass
class FrameRecord:
    client_id: str
    frame_id: str
    sent_at: float
    recv_at: float
    latency_ms: float
    resp_type: str           # "result" | "error" | "timeout" | f"unexpected:{t}"
    n_detections: Optional[int] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ClientResult:
    client_id: str
    frames: list = field(default_factory=list)
    connect_failed: Optional[str] = None
    hello_failed: Optional[str] = None


async def _run_one_client(
    client_id: str,
    url: str,
    jpeg: bytes,
    prompt: str,
    mode: str,
    num_frames: int,
    send_interval: float,
    start_delay: float,
    frame_timeout: float,
) -> ClientResult:
    result = ClientResult(client_id=client_id)
    if start_delay > 0:
        await asyncio.sleep(start_delay)
    try:
        async with websockets.connect(
            url,
            max_size=8 * 1024 * 1024,
            open_timeout=15,
        ) as ws:
            # ---- Hello / Capabilities handshake ----
            await ws.send(json.dumps({
                "type":             "hello",
                "protocol_version": 1,
                "client_id":        client_id,
                "session_id":       client_id,
            }))
            caps_raw = await asyncio.wait_for(ws.recv(), timeout=15)
            caps = json.loads(caps_raw)
            if caps.get("type") != "capabilities":
                result.hello_failed = (
                    f"expected type:'capabilities', got: {caps_raw[:200]!r}"
                )
                return result

            # ---- Send K frames; await each result before next send ----
            # Sequential per-client (1 in-flight per WS) keeps the latency
            # measurement clean: each frame's latency is its own service time
            # plus whatever queueing the other clients caused.
            for i in range(num_frames):
                if i > 0:
                    await asyncio.sleep(send_interval)
                frame_id = f"{client_id}-{i:03d}"
                header = json.dumps({
                    "type":              "frame",
                    "frame_id":          frame_id,
                    "session_id":        client_id,
                    "prompt":            prompt,
                    "generation_mode":   mode,
                    "jpeg_len":          len(jpeg),
                    "image_color_space": "RGB",
                    "image_encoding":    "jpeg",
                }).encode("utf-8")
                payload = struct.pack(">I", len(header)) + header + jpeg

                t_send = time.perf_counter()
                await ws.send(payload)

                # Drain WS frames until we see one tagged with our frame_id
                # (skip beacons; other-client frames CANNOT appear on this
                # WS — each WS is bound 1:1 to a single client task on the
                # worker side, so any mismatch IS a server-side mux bug).
                rec: Optional[FrameRecord] = None
                try:
                    async with asyncio.timeout(frame_timeout):
                        while True:
                            msg = await ws.recv()
                            obj = json.loads(msg)
                            t = obj.get("type")
                            if t == "beacon":
                                continue
                            obj_fid = obj.get("frame_id")
                            t_recv = time.perf_counter()
                            lat_ms = (t_recv - t_send) * 1000.0
                            if obj_fid != frame_id:
                                # Cross-client leakage = a real server-side bug.
                                rec = FrameRecord(
                                    client_id=client_id,
                                    frame_id=frame_id,
                                    sent_at=t_send,
                                    recv_at=t_recv,
                                    latency_ms=lat_ms,
                                    resp_type="unexpected:wrong_frame_id",
                                    error=(
                                        f"expected frame_id={frame_id!r}, "
                                        f"server delivered {obj_fid!r} "
                                        f"(type={t!r}) — WS multiplexing bug"
                                    ),
                                )
                                break
                            if t == "result":
                                rec = FrameRecord(
                                    client_id=client_id,
                                    frame_id=frame_id,
                                    sent_at=t_send,
                                    recv_at=t_recv,
                                    latency_ms=lat_ms,
                                    resp_type="result",
                                    n_detections=len(obj.get("detections", [])),
                                    raw_text=(obj.get("raw_text") or "")[:200],
                                )
                                break
                            if t == "error":
                                rec = FrameRecord(
                                    client_id=client_id,
                                    frame_id=frame_id,
                                    sent_at=t_send,
                                    recv_at=t_recv,
                                    latency_ms=lat_ms,
                                    resp_type="error",
                                    error=str(obj.get("message") or obj),
                                )
                                break
                            rec = FrameRecord(
                                client_id=client_id,
                                frame_id=frame_id,
                                sent_at=t_send,
                                recv_at=t_recv,
                                latency_ms=lat_ms,
                                resp_type=f"unexpected:{t!r}",
                                error=f"unrecognised type field; raw={obj!r}",
                            )
                            break
                except asyncio.TimeoutError:
                    rec = FrameRecord(
                        client_id=client_id,
                        frame_id=frame_id,
                        sent_at=t_send,
                        recv_at=time.perf_counter(),
                        latency_ms=frame_timeout * 1000.0,
                        resp_type="timeout",
                        error=(
                            f"no matching response within {frame_timeout}s — "
                            "possible deadlock, starvation, or worker hang"
                        ),
                    )
                assert rec is not None
                result.frames.append(rec)
    except Exception as e:
        result.connect_failed = f"{type(e).__name__}: {e}"
    return result


def _p95(xs: list[float]) -> float:
    s = sorted(xs)
    idx = int(len(s) * 0.95)
    if idx >= len(s):
        idx = len(s) - 1
    return s[idx]


async def run(args) -> int:
    jpeg = Path(args.image).read_bytes()
    if len(jpeg) < 3 or jpeg[:3] != b"\xff\xd8\xff":
        print(
            f"FAIL: image at {args.image} is not a JPEG (no SOI marker)",
            file=sys.stderr,
        )
        return 2

    print(
        f"running {args.num_clients} concurrent clients × "
        f"{args.frames_per_client} frames each "
        f"@ {args.send_interval}s interval; "
        f"client-start offset {args.start_offset}s; "
        f"per-frame timeout {args.frame_timeout}s; "
        f"fairness max-ratio {args.fairness_ratio}×",
        file=sys.stderr,
        flush=True,
    )

    if args.num_clients < 2:
        print(
            "FAIL: --num-clients must be ≥ 2; this script is meaningless "
            "with a single client.",
            file=sys.stderr,
        )
        return 2

    tasks = [
        asyncio.create_task(_run_one_client(
            client_id=f"smoke-cc-{chr(ord('A') + i)}",
            url=args.url,
            jpeg=jpeg,
            prompt=args.prompt,
            mode=args.mode,
            num_frames=args.frames_per_client,
            send_interval=args.send_interval,
            start_delay=args.start_offset * i,
            frame_timeout=args.frame_timeout,
        ))
        for i in range(args.num_clients)
    ]
    client_results: list[ClientResult] = await asyncio.gather(*tasks)

    failures: list[str] = []
    all_recs: list[FrameRecord] = []

    # ---- per-client structural checks ----
    for r in client_results:
        if r.connect_failed:
            failures.append(
                f"client {r.client_id}: WS connect failed: {r.connect_failed}"
            )
            continue
        if r.hello_failed:
            failures.append(
                f"client {r.client_id}: handshake failed: {r.hello_failed}"
            )
            continue
        if len(r.frames) != args.frames_per_client:
            failures.append(
                f"client {r.client_id}: expected {args.frames_per_client} "
                f"frame records, got {len(r.frames)} — the client task exited "
                "before sending all frames; check the WS for an early close."
            )
        for f in r.frames:
            all_recs.append(f)
            if f.resp_type == "timeout":
                failures.append(
                    f"client {r.client_id} frame {f.frame_id}: {f.error}"
                )
            elif f.resp_type == "error":
                failures.append(
                    f"client {r.client_id} frame {f.frame_id}: "
                    f"server returned type:'error' (message={f.error!r}) — "
                    "under concurrent load this usually means a lock-drain "
                    "bug or KV-cache collision"
                )
            elif f.resp_type.startswith("unexpected"):
                failures.append(
                    f"client {r.client_id} frame {f.frame_id}: {f.error}"
                )

    # ---- per-client latency aggregates over successful frames ----
    per_client_stats: dict[str, dict] = {}
    for r in client_results:
        ok_lats = [f.latency_ms for f in r.frames if f.resp_type == "result"]
        if ok_lats:
            per_client_stats[r.client_id] = {
                "n":         len(ok_lats),
                "min_ms":    round(min(ok_lats), 1),
                "median_ms": round(statistics.median(ok_lats), 1),
                "p95_ms":    round(_p95(ok_lats), 1),
                "max_ms":    round(max(ok_lats), 1),
            }

    # ---- fairness check ----
    if len(per_client_stats) >= 2:
        medians = [s["median_ms"] for s in per_client_stats.values()]
        m_min, m_max = min(medians), max(medians)
        ratio = (m_max / m_min) if m_min > 0 else float("inf")
        if ratio > args.fairness_ratio:
            failures.append(
                f"fairness violation: median latency varies by {ratio:.2f}× "
                f"across clients (min={m_min:.1f}ms, max={m_max:.1f}ms) — "
                f"exceeds fairness-ratio={args.fairness_ratio}×. "
                f"Per-client stats: {per_client_stats}. "
                "This usually indicates one client is starving the others — "
                "check whether the worker's asyncio.Lock acquire order is "
                "FIFO and whether asyncio.to_thread is queueing fairly."
            )

    # ---- timeline + stats output (always — useful even on success) ----
    print("--- per-frame timeline (chronological by send time) ---",
          file=sys.stderr)
    all_recs.sort(key=lambda r: r.sent_at)
    if all_recs:
        t0 = all_recs[0].sent_at
        for f in all_recs:
            send_rel = f.sent_at - t0
            recv_rel = f.recv_at - t0
            print(
                f"  {f.client_id} frame={f.frame_id:>16} "
                f"send=+{send_rel:6.2f}s recv=+{recv_rel:6.2f}s "
                f"lat={f.latency_ms:7.0f}ms  type={f.resp_type}"
                + (f"  detections={f.n_detections}"
                   if f.n_detections is not None else "")
                + (f"  ERROR={f.error!r}" if f.error else ""),
                file=sys.stderr,
            )
    print("--- per-client summary ---", file=sys.stderr)
    for cid in sorted(per_client_stats):
        s = per_client_stats[cid]
        print(
            f"  {cid}: n={s['n']:<2}  "
            f"min={s['min_ms']:>6}ms  "
            f"median={s['median_ms']:>6}ms  "
            f"p95={s['p95_ms']:>6}ms  "
            f"max={s['max_ms']:>6}ms",
            file=sys.stderr,
        )

    if failures:
        print(f"FAIL: {len(failures)} concurrency violations:",
              file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("OK: concurrency smoke test passed", file=sys.stderr)
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Concurrent WebSocket smoke client — see module docstring."
    )
    p.add_argument("--url",                required=True,
                   help="ws://host:port/v1/stream URL")
    p.add_argument("--image",              required=True,
                   help="path to the JPEG used as input by every frame")
    p.add_argument("--prompt",             required=True,
                   help="prompt sent in every frame header")
    p.add_argument("--mode",               default="hybrid",
                   help="generation_mode: fast | hybrid | slow")
    p.add_argument("--num-clients",        type=int,   default=2)
    p.add_argument("--frames-per-client",  type=int,   default=4)
    p.add_argument("--send-interval",      type=float, default=2.0,
                   help="seconds between consecutive sends within a client")
    p.add_argument("--start-offset",       type=float, default=0.5,
                   help="seconds between client K and K+1's first send "
                        "(staggers initial arrival so frames interleave)")
    p.add_argument("--frame-timeout",      type=float, default=120.0,
                   help="per-frame response timeout, seconds")
    p.add_argument("--fairness-ratio",     type=float, default=5.0,
                   help="max permitted ratio of per-client median latencies "
                        "(max/min); above this is treated as starvation")
    args = p.parse_args()
    rc = asyncio.run(run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
