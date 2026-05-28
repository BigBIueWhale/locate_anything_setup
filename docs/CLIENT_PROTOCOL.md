# Client protocol

The server exposes **one inference surface** and three status endpoints
over `http://127.0.0.1:8765`:

* `GET  /v1/health`       — liveness.
* `GET  /v1/capabilities` — model spec + boot calibration results.
* `GET  /v1/info`         — runtime state (GPU, arches, pixel-token examples).
* `WS   /v1/stream`       — the *only* inference path.

Everything model-related goes through `/v1/stream`. There is no
single-shot HTTP fallback — a single-frame use case simply opens the
WS, sends one Hello, sends one Frame, reads one Result, closes. The
unified surface means clients have *exactly one* protocol to implement
correctly.

---

## Endpoint reference

### `GET /v1/health`

Healthy:

```json
{
  "status":  "ok",
  "worker":  "up",
  "detail":  { "info_ok": true, "model_loaded": true }
}
```

Degraded (HTTP 503, not 200):

```json
{
  "status":  "degraded",
  "worker":  "down",
  "detail":  { "error": "deep health probe timed out after 5s; worker is wedged or asyncio loop is deadlocked" }
}
```

This is a **deep** probe — it does a real round-trip through the
Python worker's `info` path (which calls into `torch.cuda`). It
catches dead workers, deadlocked asyncio loops, and hung CUDA
drivers within a 5 s timeout. It does NOT catch a CUDA wedge that
only affects in-flight inference (the worker's `info` handler
doesn't take the model lock). See
[`docs/OPERATIONS.md`](./OPERATIONS.md#restart-policy-semantics--read-this-once)
for the recommended external monitoring pattern.

### `GET /v1/capabilities`

Full capability descriptor. Fields:

| Field | Type | Meaning |
|---|---|---|
| `protocol_version`        | int    | Bump major-only. Currently `1`. |
| `model`                   | string | `"nvidia/LocateAnything-3B"` |
| `model_dir`               | string | Path inside the container. |
| `model_manifest_sha256`   | string | SHA-256 over `(file_name, file_size)` lines — a fingerprint for the bind-mounted model directory, NOT a full content hash. For real content verification, see the per-file SHA-256 enforced at boot in `worker/validate_startup.py`. |
| `dtype`                   | string | `"bfloat16"` — the only supported dtype. |
| `attn_impl`               | string | `"flash_attention_2"`. |
| `max_image_dim`           | int    | 2240 — verified ceiling from `preprocessor_config.json`. |
| `in_token_limit`          | int    | 25600 — max ViT patches per image. |
| `max_llm_tokens_per_image`| int    | 6400. |
| `patch_px`                | int    | 14. |
| `llm_token_px`            | int    | 28. |
| `max_prompt_tokens`       | int    | 16384 — tokenizer.model_max_length. |
| `trained_generation_params` | obj  | The canonical sampling params, never changed by this server. |
| `supported_generation_modes`| list | `["fast", "hybrid", "slow"]`. |
| `supported_image_encodings` | list | `["jpeg"]`. |
| `supported_color_spaces`    | list | `["RGB"]`. |
| `calibration`             | obj    | `{n_runs, median_latency_ms, p95_latency_ms, median_fps, ...}` — measured at container boot. |
| `preset_prompts`          | obj    | Useful starting prompts (drone, household, etc.). |

Clients SHOULD pull this at startup and refuse to operate if any
expected field is missing or incompatible with what they need.

## `WebSocket /v1/stream`

Bidi, binary-aware, with frame correlation. The canonical streaming
endpoint for live operation.

### Lifecycle

```
client → server   open WS
client → server   Text   { Hello }            ← FIRST message, REQUIRED
server → client   Text   { Capabilities }     ← server confirms model spec
                  ── now ready for frames ──
client → server   Binary { Frame, payload }   ← per frame
server → client   Text   { Result | Error }   ← per frame, FIFO
server → client   Text   { Beacon }           ← ~1 Hz, advisory
```

Server emits **exactly one** `Result` or `Error` per `Frame`, in
submission order per connection. The `frame_id` in the response
equals the `frame_id` from the Frame's header — that's the only
correlation primitive. There is no `Configure` and no `Cancel`:
every Frame is self-contained (its header carries the prompt and
generation_mode), and a connection close is the only way to
abandon in-flight work.

### Hello (Text)

```json
{
  "type":             "hello",
  "protocol_version": 1,
  "client_id":        "drone-ops-01",
  "session_id":       "01HXYZ"
}
```

All four fields are required. `session_id` may be the empty string but
the field must be present. No extra keys are accepted — the deserializer
uses `#[serde(deny_unknown_fields)]`.

The server replies with a Capabilities message (same JSON as
`GET /v1/capabilities`, plus `"type": "capabilities"`). The
server's `max_inflight` is published in the periodic Beacon
message (see below), NOT in the Capabilities payload. The client
does not declare its own values.

If `protocol_version` doesn't match exactly, the server replies with
an Error frame and closes the WebSocket — there is no compatibility
downgrade.

### Frame (Binary)

```
[ 4-byte BE u32 = header_len ]
[ header_len bytes UTF-8 JSON ]
[ JPEG bytes ]
```

JSON header schema. **Every field is required** — the server's
`InferHeader` deserializer uses `#[serde(deny_unknown_fields)]` and
treats any missing field as `invalid_request`. There are no defaults
on the inference path.

```json
{
  "type":              "frame",
  "frame_id":          "f-000142",
  "session_id":        "01HXYZ-",
  "prompt":            "Point to: drone in the sky.",
  "generation_mode":   "slow",
  "jpeg_len":          300456,
  "image_color_space": "RGB",
  "image_encoding":    "jpeg"
}
```

Field rules:

* `type` — exactly `"frame"`.
* `frame_id` — non-empty string, length 1..=256. The sole
  correlation primitive between this Frame and its Result/Error.
* `session_id` — any string (may be `""`); echoed in responses for
  client-side grouping. Required to be **present** even if empty.
* `prompt` — non-empty string, length 1..=16384 characters. Should
  be one of the canonical forms in
  [`MODEL_CAPABILITIES.md`](./MODEL_CAPABILITIES.md#what-it-does-well-in-order)
  or `/v1/capabilities.preset_prompts`.
* `generation_mode` — exactly one of `"fast"`, `"hybrid"`, `"slow"`.
* `jpeg_len` — equal to the number of payload bytes after the header.
* `image_color_space` — exactly `"RGB"`.
* `image_encoding` — exactly `"jpeg"`.

The JPEG payload itself must:

* Start with `FF D8 FF`.
* Decode cleanly with libjpeg-turbo (header parse done in Rust before
  the worker is hit; full decode done in Python).
* Have both dimensions in `[32, max_image_dim=2240]`.
* Be ≤ `LA_MAX_JPEG_BYTES` (4 MiB by default).

### Result (Text)

```json
{
  "type":        "result",
  "frame_id":    "f-000142",
  "session_id":  "01HXYZ",
  "raw_text":    "<box><420><510><560><640></box>",
  "detections":  [ ... ],
  "points":      [],
  "abstained":   false,
  "image_size":  { "w": 1920, "h": 1080 },
  "resize_plan": { "dst_w": 1932, "dst_h": 1092, "n_llm_tokens": 2691, "scale": 1.006 },
  "generation_mode_used": "slow",
  "latency_ms":  812.4,
  "total_ms":    821.0
}
```

The server uses ONE JSON shape per outcome — a `result` frame does
NOT carry `ok`. Per-frame failures arrive as a separate `error`
frame (next section).

### Error (Text)

```json
{
  "type":       "error",
  "code":       400,
  "error_type": "invalid_image",
  "message":    "header.jpeg_len 100 != actual payload length 200",
  "retriable":  false,
  "frame_id":   "f-000142"
}
```

`error_type` taxonomy:

| `error_type`         | tier | retriable | meaning |
|---|---|---|---|
| `hello_required`     | conn | no  | first WS frame was not a Hello. WS closed 1008. |
| `hello_invalid`      | conn | no  | Hello JSON malformed or protocol version mismatch. WS closed 1008. |
| `invalid_request`    | frame| no  | header malformed (missing field, wrong shape, unknown key). |
| `invalid_image`      | frame| no  | JPEG malformed / dims out of `[32, max_image_dim]` / payload size mismatch. |
| `worker_unavailable` | state| **yes** | Python worker dropped its socket. Reconnect and retry. |
| `worker_protocol`    | state| no  | worker returned a frame the server couldn't decode. |
| `worker_error`       | state| yes | worker reported an internal error (e.g., CUDA OOM). Reconnect after a backoff. |
| `internal_error`     | state| yes | uncategorized server bug. |

**One JSON shape per outcome.** A `Frame` produces *either* a `type:"result"`
*or* a `type:"error"` reply — never `type:"result"` with `ok:false`. The
server unifies the Python worker's `{ok:false, ...}` responses into
`type:"error"` frames so the client only ever pattern-matches on
`type`.

### Beacon (Text, ~1 Hz, advisory)

```json
{
  "type":                   "beacon",
  "queue_depth":            1,
  "inflight":               1,
  "completed_total":        142,
  "last_completed_frame_id":"f-000141",
  "model_state":            "ready",
  "max_inflight":           2
}
```

`max_inflight` is the bounded-channel capacity for *this* WebSocket
connection on the server side. `queue_depth` plus `inflight`
≤ `max_inflight` at all times. When that sum reaches the cap, the
server's WebSocket reader stops draining the TCP receive buffer,
and the client's `send()` blocks (TCP backpressure).

The beacon is **advisory only**. A client SHOULD use it to adjust
its capture rate (e.g., paint a "GPU busy" indicator when
`inflight + queue_depth > 1`) but MUST NOT use it as a barrier — the
canonical correlation is the `frame_id` echo in `result` /
`error`.

### Backpressure

The server applies backpressure via TCP flow control:

* The Rust frontend's WebSocket reader task uses `tokio::sync::mpsc`
  with capacity equal to `max_inflight` (default 2).
* When the channel is full, the reader's `frame_tx.send().await`
  parks the task.
* The parked reader stops calling `next()` on the WebSocket
  `SplitStream`, which stops draining the TCP socket.
* TCP RWIN shrinks → the kernel-side send buffer on the client fills
  → the client's `send()` blocks (or returns `WouldBlock` for
  non-blocking I/O).

A **correctly written client** MUST respect this signal. The server
**does not drop frames** to keep up — every Frame the client sends
either gets a `result` or an `error` (in submission order).

### Reconnection

Reconnect on any WS close. The server treats every connection as
fresh:

* No state carries over the close — `session_id` is purely a client
  hint echoed in responses.
* The Hello → Capabilities handshake must be redone.
* In-flight frames are silently dropped on the close — their
  `frame_id` will never get a result. The client decides whether to
  re-send them.

Use `session_id` to thread your own client-side log of "which frames
got results vs. were dropped on reconnect."

---

## Reference client patterns

### Python WebSocket client skeleton

```python
import asyncio, json, struct, websockets

URL = "ws://127.0.0.1:8765/v1/stream"

async def run(frames):
    async with websockets.connect(URL, max_size=8 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type":             "hello",
            "protocol_version": 1,
            "client_id":        "demo",
            "session_id":       "demo-1",
        }))
        caps = json.loads(await ws.recv())
        assert caps["type"] == "capabilities", caps

        async def sender():
            for fid, jpeg in frames:
                header = json.dumps({
                    "type":              "frame",
                    "frame_id":          fid,
                    "session_id":        "demo-1",
                    "prompt":            "Point to: drone in the sky.",
                    "generation_mode":   "slow",
                    "jpeg_len":          len(jpeg),
                    "image_color_space": "RGB",
                    "image_encoding":    "jpeg",
                }).encode()
                payload = struct.pack(">I", len(header)) + header + jpeg
                await ws.send(payload)        # ← TCP backpressure here

        async def receiver():
            async for msg in ws:
                obj = json.loads(msg)
                if obj["type"] == "result":
                    handle_result(obj)
                elif obj["type"] == "error":
                    handle_error(obj)
                elif obj["type"] == "beacon":
                    update_indicator(obj)

        await asyncio.gather(sender(), receiver())
```

The reference Rust client (using `tokio-tungstenite`) follows the
same pattern. See `worker/calibration.py` for an example of how the
Python sidecar formats its own length-prefixed frames internally.
