# Client protocol

The server exposes **one inference surface** and three status endpoints
over `http://127.0.0.1:8765`:

* `GET  /v1/health`       — liveness (deep probe).
* `GET  /v1/capabilities` — model spec + boot calibration results.
* `GET  /v1/info`         — runtime state (GPU, arches, pixel-token examples).
* `WS   /v1/stream`       — the *only* inference path.

A client opens the WebSocket and immediately starts sending Frames.
There is no in-band handshake — capabilities are fetched out of band
on the HTTP endpoint above. Each WebSocket is **strictly stop-and-wait
per connection**: send one Frame, read one Result (or one Error), repeat.
Open more WebSockets to drive more concurrent inferences; the server
fans them out via a single FIFO asyncio lock around the GPU.

---

## HTTP endpoints

### `GET /v1/health`

Healthy (HTTP 200):

```json
{ "status": "ok", "worker": "up",
  "detail": { "info_ok": true, "model_loaded": true } }
```

Degraded (HTTP 503):

```json
{ "status": "degraded", "worker": "down",
  "detail": { "error": "deep health probe timed out after 5s; worker is wedged or asyncio loop is deadlocked" } }
```

This is a **deep** probe — it does a real round-trip through the Python
worker's `info` path (which calls into `torch.cuda`). It catches dead
workers, deadlocked asyncio loops, and hung CUDA drivers within a 5 s
timeout. It does NOT catch a CUDA wedge that only affects in-flight
inference (the worker's `info` handler doesn't take the model lock).
See [`docs/OPERATIONS.md`](./OPERATIONS.md) for the recommended external
monitoring pattern.

### `GET /v1/capabilities`

Model capability descriptor. Returned verbatim from the worker; fields:

| Field | Type | Meaning |
|---|---|---|
| `model`                     | string | `"nvidia/LocateAnything-3B"` |
| `model_dir`                 | string | Path inside the container. |
| `model_manifest_sha256`     | string | SHA-256 over `(file_name, file_size)` lines — a fingerprint for the bind-mounted model directory, NOT a full content hash. For real content verification see the per-file SHA-256 enforced at boot in `worker/validate_startup.py`. |
| `dtype`                     | string | `"bfloat16"` — the only supported dtype. |
| `attn_impl`                 | string | `"sdpa"` — see `docs/PINNED_VERSIONS.md`. |
| `max_image_dim`             | int    | 2240 — verified ceiling from `preprocessor_config.json`. |
| `in_token_limit`            | int    | 25600 — max ViT patches per image. |
| `max_llm_tokens_per_image`  | int    | 6400. |
| `patch_px`                  | int    | 14. |
| `llm_token_px`              | int    | 28. |
| `max_prompt_tokens`         | int    | 16384 — `tokenizer.model_max_length`. |
| `trained_generation_params` | obj    | Canonical sampling params, never changed by this server. |
| `supported_generation_modes`| list   | `["fast", "hybrid", "slow"]`. |
| `calibration`               | obj    | `{n_runs, median_latency_ms, p95_latency_ms, median_fps, ...}` measured at container boot. |
| `preset_prompts`            | obj    | Useful starting prompts (drone, household, etc.). |

Clients SHOULD pull this at startup and refuse to operate if a required
field is missing or incompatible with what they need.

### `GET /v1/info`

Live runtime metrics returned verbatim from the worker. Fields:
`ok` (worker reachable), `model_loaded`, `torch_arches`, `gpu_name`,
`gpu_total_mem_gib`, `gpu_free_mem_gib`, `gpu_used_mem_gib`,
`pixel_token_examples` (pre-computed pixel→ViT-token mappings for
common resolutions). Poll this for dashboards / autoscaling signals
— it is the only metric surface; there are no advisory messages on
the WebSocket.

---

## `WebSocket /v1/stream`

Binary, stop-and-wait per connection. One Frame in, one Result (or one
Error) out, repeat. Many clients connect concurrently; the worker
serializes GPU access through a single FIFO `asyncio.Lock`.

### Lifecycle

```
client → server   open WS
client → server   Binary { Frame }              ← per frame, length-prefixed header + JPEG
server → client   Text   { Result | Error }     ← per frame, FIFO
                  ── repeat ──
client → server   close WS                      ← clean shutdown
```

There is no in-band handshake, no advisory message, no Cancel, no
Configure. Every Frame is self-contained (its header carries the prompt
and `generation_mode`). To abandon in-flight work, close the WebSocket.

### Frame (Binary)

```
[ 4-byte BE u32 = header_len ]
[ header_len bytes UTF-8 JSON ]
[ JPEG bytes ]
```

JSON header schema. **Every field is required** — the server's
`InferHeader` deserializer uses `#[serde(deny_unknown_fields)]` and
treats any missing/extra field as a connection-fatal framing error.
There are no defaults on the inference path.

```json
{
  "frame_id":        "f-000142",
  "prompt":          "Point to: drone in the sky.",
  "generation_mode": "slow",
  "jpeg_len":        300456
}
```

Field rules:

* `frame_id` — non-empty string, length 1..=256. The sole correlation
  primitive between this Frame and its Result/Error.
* `prompt` — non-empty string, length 1..=16384 characters. Should be
  one of the canonical forms in
  [`docs/MODEL_CAPABILITIES.md`](./MODEL_CAPABILITIES.md) or
  `/v1/capabilities.preset_prompts`.
* `generation_mode` — exactly one of `"fast"`, `"hybrid"`, `"slow"`.
* `jpeg_len` — equal to the number of payload bytes after the header.

The JPEG payload itself must:

* Start with `FF D8 FF`.
* Decode cleanly with libjpeg-turbo (header parse done in Rust before
  the worker is hit; full decode done in Python).
* Be RGB (not CMYK — PIL's CMYK→RGB transform is not ICC-aware).
* Have both dimensions in `[32, max_image_dim=2240]`.
* Be ≤ `LA_MAX_JPEG_BYTES` (4 MiB by default).

### Result (Text)

```json
{
  "type":        "result",
  "frame_id":    "f-000142",
  "raw_text":    "<box><420><510><560><640></box>",
  "detections":  [],
  "points":      [],
  "abstained":   false,
  "image_size":  [1920, 1080],
  "resize_plan": { "dst_w": 1932, "dst_h": 1092, "n_llm_tokens": 2691, "scale": 1.006 },
  "generation_mode_used": "slow",
  "latency_ms":  812.4,
  "total_ms":    821.0
}
```

### Error (Text)

```json
{
  "type":      "error",
  "frame_id":  "f-000142",
  "code":      400,
  "message":   "header.prompt missing"
}
```

`code` mirrors HTTP semantics: 400-class is the client's mistake, 500-
class is the server's. The body always echoes the originating
`frame_id` so the client can correlate.

### Two error surfaces — and only two

* `type:"error"` JSON message — per-frame failure where the framing is
  still intact (prompt too long, CUDA OOM inside `.generate()`, etc.).
  The WebSocket stays open and the next Frame proceeds normally.
* WebSocket Close frame — connection-fatal: framing wrong, header JSON
  malformed, JPEG SOI absent, or the worker UDS desynced. Common
  codes: `1001` going away (server shutdown / 60 s read-idle), `1008`
  policy violation (framing/header/JPEG validation), `1011` server
  error (worker unreachable / UDS desynced). The reason names the
  condition.

### Reconnection

The server holds **no per-WebSocket state**. On any close, just open a
new WebSocket and resume sending Frames. The client owns its `frame_id`
namespace — reusing a `frame_id` across reconnects is the client's
prerogative; the server does not deduplicate.

In-flight frames are silently dropped on close — their `frame_id` will
never get a result. The client decides whether to re-send them.

### Backpressure

Per-WebSocket flow control is pure TCP. The server reads one Frame,
runs inference, writes one Result, then reads again — there is no
per-WebSocket queue. When the client sends a Frame faster than the
server processes it, the kernel-side TCP send buffer fills, and the
client's `send()` blocks (or returns `WouldBlock` on non-blocking I/O).
A correctly written client MUST respect this signal: the server **does
not drop frames** to keep up.

---

## Reference client patterns

```python
import asyncio, json, struct, urllib.request, websockets

WS_URL  = "ws://127.0.0.1:8765/v1/stream"
CAPS_URL = "http://127.0.0.1:8765/v1/capabilities"

def fetch_caps():
    with urllib.request.urlopen(CAPS_URL, timeout=10) as r:
        return json.loads(r.read())

async def run(frames, prompt):
    caps = fetch_caps()
    assert caps["model"] == "nvidia/LocateAnything-3B"
    async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:
        for fid, jpeg in frames:
            header = json.dumps({
                "frame_id":        fid,
                "prompt":          prompt,
                "generation_mode": "slow",
                "jpeg_len":        len(jpeg),
            }).encode()
            await ws.send(struct.pack(">I", len(header)) + header + jpeg)
            obj = json.loads(await ws.recv())
            if obj["type"] == "result":
                handle_result(obj)
            elif obj["type"] == "error":
                handle_error(obj)
```

The reference Rust client (using `tokio-tungstenite`) follows the same
pattern. See `examples/reference_client.py` for a full backpressure-
respecting, reconnect-with-backoff implementation.
