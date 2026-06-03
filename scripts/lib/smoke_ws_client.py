#!/usr/bin/env python3
"""
Minimal WebSocket smoke client for 04_smoke_test.sh.

Connects, sends one Frame carrying a TYPED request (A.1), awaits one
reply, and exits non-zero on any structural anomaly. Used only by the
smoke-test orchestrator — NOT a reference for clients (see
examples/reference_client.py for that).

The reply is the A.2 flat tagged union on `type`: exactly one of
`boxes` / `points` / `abstained` / `error`. This client matches that
tag and asserts on the variant — it does NOT branch on a `prompt_task`
field (that field is gone from the wire). The old "off-shape leak"
check (a point under a box task leaking into the other list) is also
gone: off-contract geometry is now either dropped per-item — surfacing
as a clean `boxes`/`points` with a non-zero `deviations_dropped` — or,
when ZERO geometry is valid for the task, an `error{model_deviation}`.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import struct
import sys
from pathlib import Path

import websockets


# The seven trained task tags (A.1 PromptRequest, snake_case). Mirrors
# rust_server/src/protocol.rs::PromptRequest and worker/prompts.py::req_*.
TASKS = ("detection", "phrase_single", "phrase_multi",
         "text_grounding", "scene_text", "gui_box", "point")

# A box-shaped task yields the `boxes` variant; `point` yields `points`.
# (scene_text/detection/phrase_*/text_grounding/gui_box are all box-shaped.)
BOX_TASKS = {"detection", "phrase_single", "phrase_multi",
             "text_grounding", "scene_text", "gui_box"}

# The success variants (everything but `error`).
SUCCESS_TYPES = ("boxes", "points", "abstained")


def build_request(args) -> dict:
    """Assemble the typed `request` object (A.1) from the CLI flags.

    Tagged on `task`; each task carries exactly its own slot. The server
    validates the slot and compiles the trained prompt — we only build the
    right shape here."""
    task = args.task
    if task == "detection":
        if not args.categories:
            print("FAIL: --task detection requires --categories a,b,c",
                  file=sys.stderr)
            sys.exit(2)
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        if not cats:
            print("FAIL: --categories produced zero non-empty entries",
                  file=sys.stderr)
            sys.exit(2)
        return {"task": "detection", "categories": cats}
    if task in ("phrase_single", "phrase_multi", "point"):
        if args.phrase is None:
            print(f"FAIL: --task {task} requires --phrase", file=sys.stderr)
            sys.exit(2)
        return {"task": task, "phrase": args.phrase}
    if task == "text_grounding":
        if args.text is None:
            print("FAIL: --task text_grounding requires --text",
                  file=sys.stderr)
            sys.exit(2)
        return {"task": "text_grounding", "text": args.text}
    if task == "gui_box":
        if args.description is None:
            print("FAIL: --task gui_box requires --description",
                  file=sys.stderr)
            sys.exit(2)
        return {"task": "gui_box", "description": args.description}
    if task == "scene_text":
        return {"task": "scene_text"}
    print(f"FAIL: unknown --task {task!r}", file=sys.stderr)
    sys.exit(2)


async def run(args):
    jpeg = Path(args.image).read_bytes()
    if len(jpeg) < 3 or jpeg[0:3] != b"\xff\xd8\xff":
        print(f"FAIL: image at {args.image} is not a JPEG (SOI absent)",
              file=sys.stderr)
        sys.exit(2)

    request = build_request(args)

    print(f"connecting to {args.url}", flush=True)
    async with websockets.connect(
        args.url,
        max_size=8 * 1024 * 1024,
        open_timeout=15,
    ) as ws:
        # A.1 InferHeader: {frame_id, request, generation_mode, jpeg_len}.
        header = json.dumps({
            "frame_id":        "smoke-001",
            "request":         request,
            "generation_mode": args.mode,
            "jpeg_len":        len(jpeg),
        }).encode("utf-8")
        payload = struct.pack(">I", len(header)) + header + jpeg
        await ws.send(payload)
        print(f"sent frame (request={json.dumps(request)}, "
              f"header={len(header)} bytes, jpeg={len(jpeg)} bytes)",
              flush=True)

        # Await the single reply keyed by our frame_id. A.2 is a flat
        # tagged union: boxes XOR points XOR abstained XOR error — there
        # are no control / advisory messages on the WS, so any other tag
        # is a real protocol bug, not something to skip. A WS Close from
        # the server (1001/1008/1011) is a documented connection-fatal
        # outcome — surface it as a clean FAIL line so CI logs aren't
        # polluted by a Python stack trace.
        reply = None
        try:
            async with asyncio.timeout(args.timeout):
                while True:
                    msg = await ws.recv()
                    obj = json.loads(msg)
                    t = obj.get("type")
                    fid = obj.get("frame_id")
                    if fid != "smoke-001":
                        print(f"FAIL: reply for wrong frame_id={fid!r} "
                              f"(type={t!r}): {obj}", file=sys.stderr)
                        sys.exit(5)
                    if t == "error":
                        print(f"FAIL: server returned per-frame error "
                              f"(code={obj.get('code')!r}): {obj}",
                              file=sys.stderr)
                        sys.exit(4)
                    if t in SUCCESS_TYPES:
                        reply = obj
                        break
                    print(f"FAIL: unexpected reply type={t!r} "
                          f"(not one of {SUCCESS_TYPES} / error): {obj}",
                          file=sys.stderr)
                    sys.exit(5)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"FAIL: server closed the WebSocket before responding: "
                  f"code={getattr(e, 'code', None)} "
                  f"reason={getattr(e, 'reason', None)!r}",
                  file=sys.stderr)
            sys.exit(10)

        if reply is None:
            print("FAIL: no reply received", file=sys.stderr)
            sys.exit(6)

        rtype = reply["type"]

        # ---- Meta fields are present on EVERY success variant (A.2 Meta) ----
        if not isinstance(reply.get("raw_text"), str):
            print(f"FAIL: reply.raw_text is not a string: {reply}",
                  file=sys.stderr)
            sys.exit(8)
        if not isinstance(reply.get("latency_ms"), (int, float)):
            print(f"FAIL: reply.latency_ms is missing or not numeric: {reply}",
                  file=sys.stderr)
            sys.exit(9)
        if not isinstance(reply.get("model_output_truncated"), bool):
            print(f"FAIL: reply.model_output_truncated is missing or "
                  f"not bool: {reply}", file=sys.stderr)
            sys.exit(12)
        deviations = reply.get("deviations_dropped")
        if not isinstance(deviations, int) or isinstance(deviations, bool):
            print(f"FAIL: reply.deviations_dropped is missing or not an int: "
                  f"{reply}", file=sys.stderr)
            sys.exit(11)

        # ---- Geometry list is the one that matches the variant, with the
        #      required `label` on each item (A.2 LabeledBox/LabeledPoint).
        #      The abstained variant carries NEITHER list. There is no
        #      cross-list off-shape leak to check any more — a box-shaped
        #      reply is `boxes` only, a point reply is `points` only.
        n_geom = 0
        if rtype == "boxes":
            boxes = reply.get("boxes")
            if not isinstance(boxes, list):
                print(f"FAIL: boxes reply has no boxes list: {reply}",
                      file=sys.stderr)
                sys.exit(7)
            for b in boxes:
                if not isinstance(b.get("label"), str):
                    print(f"FAIL: box missing required string label: {b}",
                          file=sys.stderr)
                    sys.exit(7)
                if not (isinstance(b.get("bbox_px"), list)
                        and len(b["bbox_px"]) == 4):
                    print(f"FAIL: box bbox_px is not a 4-list: {b}",
                          file=sys.stderr)
                    sys.exit(7)
            n_geom = len(boxes)
        elif rtype == "points":
            points = reply.get("points")
            if not isinstance(points, list):
                print(f"FAIL: points reply has no points list: {reply}",
                      file=sys.stderr)
                sys.exit(7)
            for p in points:
                if not isinstance(p.get("label"), str):
                    print(f"FAIL: point missing required string label: {p}",
                          file=sys.stderr)
                    sys.exit(7)
                if not (isinstance(p.get("point_px"), list)
                        and len(p["point_px"]) == 2):
                    print(f"FAIL: point point_px is not a 2-list: {p}",
                          file=sys.stderr)
                    sys.exit(7)
            n_geom = len(points)
        # rtype == "abstained": no geometry list at all — nothing to read.

        # ---- Variant assertion. The smoke caller declares the variant it
        #      expects this task+image to produce. `boxes`/`points`/`abstained`
        #      are all "success"; --expect-type pins down which.
        if args.expect_type and rtype != args.expect_type:
            print(f"FAIL: reply type={rtype!r} did not match the smoke "
                  f"caller's --expect-type={args.expect_type!r}: {reply}",
                  file=sys.stderr)
            sys.exit(15)

        # ---- model_output_truncated diagnostic. The synthetic calibration
        #      image is small; a truncated reply (max_new_tokens hit) on it is
        #      a strong signal of a degenerate/looping decode — surface it, but
        #      it is non-fatal (the partial geometry is still valid).
        print(f"OK: type={rtype}, latency={reply['latency_ms']} ms, "
              f"{n_geom} {('points' if rtype == 'points' else 'boxes')}, "
              f"deviations_dropped={deviations}, "
              f"truncated={reply['model_output_truncated']}",
              flush=True)
        truncated = reply.get("model_output_truncated", "<absent>")
        print(f"raw_text={reply['raw_text'][:200]!r} "
              f"(model_output_truncated={truncated})", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Typed-request WebSocket smoke client — see module docstring."
    )
    p.add_argument("--url",    required=True)
    p.add_argument("--image",  required=True)
    # Typed request (A.1): --task picks the slot flag that applies.
    p.add_argument("--task", required=True, choices=TASKS,
                   help="trained task tag (detection→--categories, "
                        "phrase_single/phrase_multi/point→--phrase, "
                        "text_grounding→--text, gui_box→--description, "
                        "scene_text→no slot)")
    p.add_argument("--categories",
                   help="detection only: comma-separated category list (1..=10)")
    p.add_argument("--phrase",
                   help="phrase_single / phrase_multi / point slot")
    p.add_argument("--text",        help="text_grounding slot")
    p.add_argument("--description", help="gui_box slot")
    p.add_argument("--mode",   default="hybrid")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--expect-type", default=None, choices=SUCCESS_TYPES,
                   help="Optional: assert the RESPONSE variant equals this "
                        "exact A.2 success tag (one of 'boxes'/'points'/"
                        "'abstained'). Smoke tests use this to verify the "
                        "task+image produced the expected variant. (Replaces "
                        "the old --expect-task, which asserted on the removed "
                        "prompt_task wire field.)")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
