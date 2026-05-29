#!/usr/bin/env python3
"""
Minimal WebSocket smoke client for 04_smoke_test.sh.

Connects, sends one Frame, awaits one Result, exits non-zero on any
structural anomaly. Used only by the smoke-test orchestrator — NOT a
reference for clients (see examples/reference_client.py for that).
"""

from __future__ import annotations
import argparse
import asyncio
import json
import struct
import sys
from pathlib import Path

import websockets


async def run(args):
    jpeg = Path(args.image).read_bytes()
    if len(jpeg) < 3 or jpeg[0:3] != b"\xff\xd8\xff":
        print(f"FAIL: image at {args.image} is not a JPEG (SOI absent)",
              file=sys.stderr)
        sys.exit(2)

    print(f"connecting to {args.url}", flush=True)
    async with websockets.connect(
        args.url,
        max_size=8 * 1024 * 1024,
        open_timeout=15,
    ) as ws:
        header = json.dumps({
            "frame_id":        "smoke-001",
            "prompt":          args.prompt,
            "generation_mode": args.mode,
            "jpeg_len":        len(jpeg),
        }).encode("utf-8")
        payload = struct.pack(">I", len(header)) + header + jpeg
        await ws.send(payload)
        print(f"sent frame (header={len(header)} bytes, jpeg={len(jpeg)} bytes)",
              flush=True)

        # Await the single response keyed by our frame_id. Only
        # result / error are valid on the WS in this protocol — any
        # other type is a real protocol bug, not something to skip.
        # A WS Close from the server (1001/1008/1011) is a documented
        # connection-fatal outcome — surface it as a clean FAIL line so
        # CI logs aren't polluted by a Python stack trace.
        result = None
        try:
            async with asyncio.timeout(args.timeout):
                while True:
                    msg = await ws.recv()
                    obj = json.loads(msg)
                    t = obj.get("type")
                    fid = obj.get("frame_id")
                    if t == "result" and fid == "smoke-001":
                        result = obj
                        break
                    elif t == "error" and fid == "smoke-001":
                        print(f"FAIL: server returned per-frame error: {obj}",
                              file=sys.stderr)
                        sys.exit(4)
                    else:
                        print(f"FAIL: unexpected message (type={t!r}, "
                              f"frame_id={fid!r}): {obj}", file=sys.stderr)
                        sys.exit(5)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"FAIL: server closed the WebSocket before responding: "
                  f"code={getattr(e, 'code', None)} "
                  f"reason={getattr(e, 'reason', None)!r}",
                  file=sys.stderr)
            sys.exit(10)
        # Structural assertions on the result.
        if result is None:
            print("FAIL: no result received", file=sys.stderr)
            sys.exit(6)
        if not isinstance(result.get("detections"), list):
            print(f"FAIL: result.detections is not a list: {result}",
                  file=sys.stderr)
            sys.exit(7)
        if not isinstance(result.get("raw_text"), str):
            print(f"FAIL: result.raw_text is not a string: {result}",
                  file=sys.stderr)
            sys.exit(8)
        if not isinstance(result.get("latency_ms"), (int, float)):
            print(f"FAIL: result.latency_ms is missing or not numeric: {result}",
                  file=sys.stderr)
            sys.exit(9)
        print(f"OK: latency={result['latency_ms']} ms, "
              f"{len(result['detections'])} boxes, "
              f"{len(result.get('points', []))} points, "
              f"abstained={result.get('abstained')}",
              flush=True)
        print(f"raw_text={result['raw_text'][:200]!r}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url",    required=True)
    p.add_argument("--image",  required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--mode",   default="hybrid")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
