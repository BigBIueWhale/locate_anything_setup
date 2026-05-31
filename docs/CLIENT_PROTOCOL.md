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
| `prompt_templates_reference_url` | string | URL of the single source of truth for allowed prompt templates (currently [`worker/prompts.py`](../worker/prompts.py) on GitHub). Also embedded verbatim in every per-frame prompt-validation rejection diagnostic so the client always knows where to look. |
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
* `prompt` — non-empty string, length 1..=16384 characters. **MUST**
  exactly conform to one of the seven canonical LocateAnything-3B
  prompt templates. The single source of truth for those templates is
  [`worker/prompts.py`](../worker/prompts.py); the same URL is carried
  in `/v1/capabilities.prompt_templates_reference_url` and is echoed in
  every rejection diagnostic. Strict enforcement happens at the
  WebSocket edge by the Rust validator
  [`rust_server/src/prompt_validator.rs`](../rust_server/src/prompt_validator.rs)
  — no regex, byte-exact, per-word strict (including the `matches` /
  `match` asymmetry, the literal `</c>` category separator, and the
  trailing `.`). Anything that does not exactly match a canonical
  template is rejected with a per-frame `type:"error"` describing
  what failed, the closest canonical template, optional English-language
  heuristic hints (e.g. `Point at:` → suggest `Point to:`), and the
  reference URL. The connection stays open after a rejection; the
  client can correct and send the next Frame normally.
* `generation_mode` — exactly one of `"fast"`, `"hybrid"`, `"slow"`.
* `jpeg_len` — equal to the number of payload bytes after the header.

The JPEG payload itself must satisfy the *trained-correct contract* —
every constraint here is what NVIDIA's training pipeline assumed, and
violations are rejected at the network edge with a Close 1008 so the
client knows immediately that the model would have seen a different
input than it sent:

* **Start with `FF D8 FF`** — JPEG SOI marker.
* **Decode cleanly with libjpeg-turbo** (header parse done in Rust;
  full decode done in Python).
* **Be RGB**, not CMYK. PIL's CMYK→RGB transform isn't ICC-aware and
  produces wrong colours; we reject CMYK rather than degrade.
* **Both dimensions in `[32, max_image_dim=2240]`** px.
* **`(W // 14) × (H // 14) ≤ 25,600`** ViT patches (the exact formula
  the model's `image_processing_locateanything.py:52` checks against
  `in_token_limit=25,600`). At the 2240 px-per-side cap a square input
  has 160 × 160 = 25,600 patches — right at the budget — so any input
  meaningfully larger than 2240×2240 *(or non-square with one dim ≥ 2254)*
  would trigger the preprocessor's internal BICUBIC downscale. We refuse
  rather than scale, so the client's `frame_id` corresponds to the
  spatial frame the model actually saw.
* **`(W // 14) < 512` AND `(H // 14) < 512`** patches per side — the
  MoonViT positional embedding is a 64×64 base learnable embedding
  bicubic-interpolated to the runtime grid; 512 patches per side is
  the documented "Exceed pos emb" hard cap (line 68 of the same file).
  Equivalently each dim < 7168 px.
* **Be ≤ `LA_MAX_JPEG_BYTES`** (4 MiB by default).
* **If the JPEG carries an ICC profile, it should be sRGB** — but you
  do not have to strip it. The worker uses `PIL.ImageCms.profileToProfile`
  with `PERCEPTUAL` rendering intent to colour-manage any non-sRGB
  profile (Adobe-RGB, Display-P3, ProPhoto, etc.) into sRGB before the
  model sees it. A genuinely corrupt or unsupported profile is rejected
  rather than silently fall back to the colour shift.

In short, **strict in, faithful out**. The client never has to ask
"what did the model actually see?" — if the request reached the model,
it saw exactly the pixels you sent, modulo a BICUBIC resize to the next
`merge_kernel_size × patch_size` = 28-px multiple on each axis (which
is the trained preprocessing path and is mathematically identical to
what NVIDIA does at training time).

### Result (Text)

```json
{
  "type":        "result",
  "frame_id":    "f-000142",
  "raw_text":    "<ref>drone</ref><box><420><510><560><640></box><|im_end|>",
  "detections":  [{"label": "drone", "bbox_norm": [420,510,560,640], "bbox_px": [806.4, 550.8, 1075.2, 691.2]}],
  "points":      [],
  "abstained":   false,
  "model_output_truncated": false,
  "prompt_task": "detection",
  "image_size":  [1920, 1080],
  "resize_plan": { "dst_w": 1932, "dst_h": 1092, "n_llm_tokens": 2691, "scale": 1.006 },
  "generation_mode_used": "slow",
  "latency_ms":  812.4,
  "total_ms":    821.0
}
```

`prompt_task` is the wire name of the canonical template the server
classified the prompt as — one of `"detection"`, `"phrase_single"`,
`"phrase_multi"`, `"text_grounding"`, `"scene_text"`, `"gui_box"`,
`"point"`. The mapping to response shape is fixed by NVIDIA's
training: `"point"` → 2-coord points in `points[]` (with `detections`
guaranteed empty); every other value → 4-coord boxes in `detections[]`
(with `points` guaranteed empty). The server enforces this by filtering
off-shape model output before stamping the response, so a naive client
can branch on `prompt_task` alone without inspecting both lists. Any
off-shape model output is dropped (zero cross-shape events were
observed in 3,444 trials at trained sampling params spanning all 7
templates × adversarial prompts; the filter is therefore a forward-
compat guard rail rather than a frequent rejection path).

`raw_text` carries the full decoded model output including structural
tokens (`<box>`, `<ref>`, `<0>`..`<1000>`) and the trailing
`<|im_end|>` end-of-turn marker. The model's custom `.generate()` loop
at `models/LocateAnything-3B/modeling_locateanything.py:464,500-501`
terminates exclusively on `<|im_end|>` OR `max_new_tokens=8192`
exhaustion — these are the ONLY two exit reasons.

`model_output_truncated: true` exactly captures the budget-exhaustion
case (`not raw_text.endswith("<|im_end|>")`). When it fires, the
response is necessarily incomplete — any block whose closing `</box>`
did not fit is silently dropped at the parser layer. Partial blocks
are dropped to match NVIDIA's own evaluation behaviour
(`Embodied/evaluation/inference_grounding_ddp.py:282-300` requires
the full 4-coord shape; partials never enter their metrics either).
Truncation is empirically rare on the trained-budget
`max_new_tokens=8192` (max observed output in the live drone domain:
~50 tokens, 99.4 % headroom; the only realistic trigger is dense-scene
scene-text detection on hundreds of distinct text regions, ~273 s in
slow mode).

Per-item shape inside `detections` / `points`:

```json
{ "label": "drone" | null, "bbox_norm": [x1, y1, x2, y2], "bbox_px": [x1, y1, x2, y2] }   // detections[i]
{ "label": "drone in the sky" | null, "point_norm": [x, y], "point_px": [x, y] }            // points[i]
```

`label` is the string captured by the preceding `<ref>...</ref>` tag,
inherited across all sibling `<box>` blocks of the same ref-run (so a
template-3 multi-instance grounding query `Locate all the instances
that match the following description: PHRASE.` returns N detections
all labeled `PHRASE`, mirroring NVIDIA's eval-time parser at
`Embodied/evaluation/inference_grounding_ddp.py:379-427`). `label`
is `null` only for bare `<box>` blocks the model emitted without
a preceding `<ref>` (off-pattern; none of the seven canonical
templates trains this shape). Coords are integers in `[0, 1000]`
for `*_norm` and floats in source-image pixels for `*_px`.

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
  still intact (prompt validation rejection, JPEG malformed, image
  dimensions out of range, generation_mode unknown, etc.). For most
  per-frame errors the WebSocket stays open and the next Frame
  proceeds normally. **Two exceptions**: `code:504` (inference
  timeout — exceeded `LA_INFERENCE_TIMEOUT_S`, default 600 s) and
  `code:500` with `message` starting `"CUDA out of memory"` are
  followed *immediately* by a `Close(1011)` because the worker
  self-exits to restore CUDA state from scratch. The typed error
  reaches the client first (the worker awaits the UDS write before
  calling `os._exit`), then the Close. Reconnect after ~10–15 s
  (model reload + boot self-test); see
  [`docs/OPERATIONS.md`](./OPERATIONS.md#worker-self-exit-on-timeout--oom).
* WebSocket Close frame — connection-fatal: framing wrong, header JSON
  malformed, JPEG SOI absent, the worker UDS desynced, or the worker
  self-exited per the rule above. Common codes: `1001` going away
  (server shutdown / 60 s read-idle), `1008` policy violation
  (framing/header/JPEG validation), `1011` server error (worker
  unreachable / worker self-exited for restart). The reason names
  the condition.

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
