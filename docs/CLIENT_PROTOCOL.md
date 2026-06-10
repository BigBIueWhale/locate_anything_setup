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
| `prompt_templates_reference_url` | string | URL of the single source of truth for the seven trained prompt templates (currently [`worker/prompts.py`](../worker/prompts.py) on GitHub). Also embedded verbatim in every per-frame slot-validation rejection diagnostic so the client always knows where to look. |
| `preset_prompts`            | obj    | Useful starting requests (drone, household, etc.), as **typed** `{label, request, generation_mode}` objects — see below. |

Clients SHOULD pull this at startup and refuse to operate if a required
field is missing or incompatible with what they need.

#### `preset_prompts` — typed requests, not strings

Because a Frame carries a **typed** `request` (not a free-form prompt
string), `preset_prompts` advertises typed requests too. It is an object
of named bundles (`drone_ranked`, `household`, …); each bundle is a list
of `{label, request, generation_mode}` objects whose `request` is a
ready-to-send `PromptRequest` and whose `generation_mode` is the mode
recommended for that task. A client drops a preset's `request` and
`generation_mode` straight into a Frame header, unchanged:

```json
{
  "drone_ranked": [
    { "label": "drone (point)",  "request": { "task": "point",     "phrase": "drone in the sky" },          "generation_mode": "slow" },
    { "label": "drone (detect)", "request": { "task": "detection",  "categories": ["drone"] },               "generation_mode": "slow" }
  ],
  "household": [
    { "label": "office",         "request": { "task": "detection",  "categories": ["bottle","cup","laptop","keyboard","mouse","monitor","book"] }, "generation_mode": "hybrid" }
  ]
}
```

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
Configure. Every Frame is self-contained (its header carries the typed
`request` and `generation_mode`). To abandon in-flight work, close the
WebSocket.

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
  "request":         { "task": "point", "phrase": "drone in the sky" },
  "generation_mode": "slow",
  "jpeg_len":        300456
}
```

Field rules:

* `frame_id` — non-empty string, length 1..=256. The sole correlation
  primitive between this Frame and its Result/Error.
* `request` — a **typed** inference request: a sum type internally
  tagged on `task`, with exactly one variant per trained template (the
  seven tasks below). You never type a prompt string and you never type
  the trained scaffolding (`Locate all the instances that…`, the `</c>`
  category separator, the trailing `.`, the `matches` / `match`
  asymmetry). The server **compiles** your typed request into the exact
  trained prompt string via
  [`rust_server/src/prompt_validator.rs`](../rust_server/src/prompt_validator.rs)
  — byte-equal to [`worker/prompts.py`](../worker/prompts.py), the single
  source of truth (boot-time drift-checked against the Rust constants).
* `generation_mode` — exactly one of `"fast"`, `"hybrid"`, `"slow"`.
  Serde rejects any other string at parse time (Close 1008) — there is
  no client-side enum to keep in sync, no default, and no way to send a
  fourth mode by accident.
* `jpeg_len` — equal to the number of payload bytes after the header.

#### The typed `request` — the seven tasks

`request` is tagged on `task`. Each variant carries exactly the slots
that task needs, and **only** those slots (`deny_unknown_fields` per
variant). Illegal states are unrepresentable: a wrong slot for a task, a
stray field, or an unknown `task` value is rejected at parse time
(Close 1008), before any inference runs.

| `task` | Slots | Compiles to (trained prompt) |
|---|---|---|
| `detection`      | `categories: [string]` (1..=10) | `Locate all the instances that matches the following description: ` + `categories` joined by `</c>` + `.` |
| `phrase_single`  | `phrase: string`      | `Locate a single instance that matches the following description: ` + phrase + `.` |
| `phrase_multi`   | `phrase: string`      | `Locate all the instances that match the following description: ` + phrase + `.` (note: `match`) |
| `text_grounding` | `text: string`        | `Please locate the text referred as ` + text + `.` |
| `scene_text`     | *(none)*              | `Detect all the text in box format.` (literal; no slot, no appended `.`) |
| `gui_box`        | `description: string` | `Locate the region that matches the following description: ` + description + `.` |
| `point`          | `phrase: string`      | `Point to: ` + phrase + `.` |

The `</c>` join, the `matches` / `match` asymmetry, and the trailing `.`
are **server-internal**: the client only ever supplies the raw slot
values. Wire examples:

```json
{ "frame_id":"f-1", "request":{ "task":"detection", "categories":["drone","bird"] }, "generation_mode":"slow",   "jpeg_len":300456 }
{ "frame_id":"f-2", "request":{ "task":"point",     "phrase":"drone in the sky" },   "generation_mode":"slow",   "jpeg_len":300456 }
{ "frame_id":"f-3", "request":{ "task":"scene_text" },                                "generation_mode":"hybrid", "jpeg_len":300456 }
{ "frame_id":"f-4", "request":{ "task":"phrase_multi", "phrase":"people wearing hats" }, "generation_mode":"hybrid", "jpeg_len":300456 }
{ "frame_id":"f-5", "request":{ "task":"text_grounding", "text":"STOP" },             "generation_mode":"slow",   "jpeg_len":300456 }
{ "frame_id":"f-6", "request":{ "task":"gui_box", "description":"the search button" },"generation_mode":"fast",   "jpeg_len":300456 }
```

#### Slot rules (server-enforced on the typed slots)

Slot strings are validated on the typed values, before the prompt is
assembled. The server **rejects** a bad slot — it never silently
normalizes, strips, or drops it, so the compiled prompt is exactly what
you supplied. Each slot string (`phrase`, `text`, `description`, and each
`category`) must be:

* **non-empty**;
* in **Unicode NFC** (rejected, never normalized, if not already NFC);
* free of **control characters** (this also excludes interior
  newlines/tabs);
* free of **leading/trailing whitespace** (slots carry no padding);
* free of the literal **`</c>`** (the structural category separator is
  server-inserted only);
* **not ending in `.`** (the server appends the single trailing period);
* **≤ 200 characters**.

Per-task extras: `detection.categories` is **1..=10** entries (each a
slot string as above) and **no category may contain a comma**;
`point.phrase` **may not contain a comma** (NVIDIA's pointing eval issues
one `Point to: <single>.` per category and merges client-side — send one
Frame per category). Free-form phrases (`phrase_single`, `phrase_multi`)
*may* contain commas. The assembled prompt is separately capped at
16384 characters.

A slot violation yields a per-frame `type:"error"` with
`code:"invalid_request"` whose `message` names exactly what failed, the
canonical template it belongs to, and the reference URL. The connection
stays open after a rejection; the client corrects the slot and sends the
next Frame normally.

The JPEG payload itself must satisfy the *trained-correct contract* —
every constraint here is what NVIDIA's training pipeline assumed, and
violations are rejected at the network edge with a Close 1008 so the
client knows immediately that the model would have seen a different
input than it sent:

* **Start with `FF D8 FF`** — JPEG SOI marker.
* **Decode cleanly with libjpeg-turbo** (header parse done in Rust;
  full decode done in Python).
* **Be RGB**, not CMYK. PIL's CMYK→RGB transform isn't ICC-aware (does not honor embedded ICC colour profiles) and
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

### Response (Text) — a strict tagged union on `type`

The reply to a Frame is **exactly one** of four variants, tagged on
`type` ∈ {`boxes`, `points`, `abstained`, `error`} — mutually exclusive
and exhaustive. The Rust edge deserializes the worker's reply into this
union (`deny_unknown_fields` per variant), so a drifted worker field
fails loud at the edge rather than reaching the client.

> **The wire is a strict sum type with nothing optional, and illegal
> states are unrepresentable. A client does ZERO defensive parsing: it
> `match`es on `type` and reads exactly the fields that variant
> guarantees.** There is no "is this field present?", no "are both lists
> empty?", no "is `label` maybe null?". `boxes` always has a non-empty
> `boxes` array of labeled boxes; `points` always has labeled points;
> `abstained` carries only metadata; `error` carries only `code` +
> `message`. Strict-at-the-edge is what makes every downstream consumer
> simpler — the validation happens once, at the server boundary.

**`boxes`** — the model produced ≥1 valid labeled box (every box-shaped
task: `detection`, `phrase_single`, `phrase_multi`, `text_grounding`,
`scene_text`, `gui_box`):

```json
{
  "type":        "boxes",
  "frame_id":    "f-000142",
  "boxes":       [ { "label": "drone", "bbox_norm": [420,510,560,640], "bbox_px": [806.4, 550.8, 1075.2, 691.2] } ],
  "raw_text":    "<ref>drone</ref><box><420><510><560><640></box><|im_end|>",
  "model_output_truncated": false,
  "deviations_dropped": 0,
  "image_size":  [1920, 1080],
  "resize_plan": { "dst_w": 1932, "dst_h": 1092, "n_llm_tokens": 2691, "scale": 1.006 },
  "generation_mode_used": "slow",
  "latency_ms":  812.4,
  "total_ms":    821.0
}
```

**`points`** — the model produced ≥1 valid labeled point (only the
`point` task):

```json
{
  "type":        "points",
  "frame_id":    "f-000143",
  "points":      [ { "label": "drone in the sky", "point_norm": [500,300], "point_px": [960.0, 324.0] } ],
  "raw_text":    "<ref>drone in the sky</ref><box><500><300></box><|im_end|>",
  "model_output_truncated": false,
  "deviations_dropped": 0,
  "image_size":  [1920, 1080],
  "resize_plan": { "dst_w": 1932, "dst_h": 1092, "n_llm_tokens": 2691, "scale": 1.006 },
  "generation_mode_used": "slow",
  "latency_ms":  640.2,
  "total_ms":    648.9
}
```

**`abstained`** — the model produced **no geometry at all** (all
`<box>None</box>` / empty); carries metadata only, no geometry array:

```json
{
  "type":        "abstained",
  "frame_id":    "f-000144",
  "raw_text":    "<ref>drone</ref><box>None</box><|im_end|>",
  "model_output_truncated": false,
  "deviations_dropped": 0,
  "image_size":  [1920, 1080],
  "resize_plan": { "dst_w": 1932, "dst_h": 1092, "n_llm_tokens": 2691, "scale": 1.006 },
  "generation_mode_used": "hybrid",
  "latency_ms":  205.1,
  "total_ms":    211.7
}
```

**`error`** — a per-frame failure; carries only `code` + `message`:

```json
{
  "type":      "error",
  "frame_id":  "f-000145",
  "code":      "invalid_request",
  "message":   "pointing / GUI grounding (point output) phrase contains ','. NVIDIA's pointing eval calls `Point to: <single>.` once per category and merges client-side; send one Frame per category and merge on the client. …"
}
```

#### `boxes` vs `points` vs `abstained` — what each guarantees

The shape is fixed by the task: the `point` task yields `points`, every
other task yields `boxes`. The model *cannot* return both lists, and the
server never returns an empty geometry array on a `boxes`/`points`
variant — zero geometry routes to `abstained` instead (see below). So
the client branches on `type` and reads exactly one array.

**`abstained` is a VARIANT, not a boolean flag on a result.** It means
"the model produced no parseable geometry" — every `<box>None</box>` or
empty output — and deliberately collapses "all categories abstained"
with "model emitted nothing usable to render"; both are "nothing to
draw" from the client's perspective. It is **not** a calibrated
confidence — treat it as "no usable output", not "the model is confident
there is nothing there."

`deviations_dropped` (in the metadata of `boxes`/`points`/`abstained`)
counts **off-contract items the server dropped while keeping the valid
geometry**: output in the wrong shape for the task (a point on a
box-shaped task, or a box on the `point` task) and any box the model
emitted with no `<ref>` label. (On the `point` task the model's trained
output is a *bare* `<box><x><y></box>` with no `<ref>`; the server labels
each such point with the queried phrase and returns it — these are **not**
dropped.) It is usually `0` and is **non-fatal** —
co-emitted valid geometry is still returned in the variant's array. The
dropped tokens remain verbatim in `raw_text`. Zero cross-shape events
were observed in 3,444 trials at trained sampling params spanning all 7
templates × adversarial prompts; this is a forward-compat guard rail
rather than a frequent path.

A frame becomes an **`error{code:"model_deviation"}`** (not `abstained`,
not `boxes`/`points`) only when the model **did** emit geometry but
**zero** of it was valid for the task — all off-shape or all unlabeled —
or the decoded output is unparseable gibberish. In that case there is
nothing usable to return, so the frame is a loud per-frame error rather
than a misreported abstention. The distinction is exact:

* no geometry emitted → `abstained`;
* some geometry emitted, ≥1 valid → `boxes`/`points` (off-contract
  surplus counted in `deviations_dropped`);
* geometry emitted, zero valid → `error{code:"model_deviation"}`.

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

Per-item shape inside `boxes[]` (the `boxes` variant) / `points[]` (the
`points` variant):

```json
{ "label": "drone", "bbox_norm": [x1, y1, x2, y2], "bbox_px": [x1, y1, x2, y2] }   // boxes[i]
{ "label": "drone in the sky", "point_norm": [x, y], "point_px": [x, y] }          // points[i]
```

`label` is **required** — every box and every point carries one. It is
the string captured by the preceding `<ref>...</ref>` tag, inherited
across all sibling `<box>` blocks of the same ref-run (so a
multi-instance grounding query `request:{task:"phrase_multi",phrase:…}`
returns N boxes all labeled with that phrase, mirroring NVIDIA's
eval-time parser at
`Embodied/evaluation/inference_grounding_ddp.py:379-427`). A bare
`<box>` block the model emits without a preceding `<ref>` has no label
and is **off-contract**: it is dropped and counted in
`deviations_dropped` (never returned as a `label:null` item) — `label`
is never null on the wire, so the type guarantees you a string. Coords
are integers in `[0, 1000]` for `*_norm` (canonical: clamped to the grid
and, for boxes, corner-sorted so `x1<=x2`, `y1<=y2`) and floats in
source-image pixels for `*_px`.

### Error codes

The `error` variant's `code` is one of **four string** values (never a
numeric HTTP-style code). The body always echoes the originating
`frame_id` so the client can correlate, and `message` is an
information-generous English diagnostic.

| `code` | Meaning |
|---|---|
| `"invalid_request"` | A slot failed validation (empty, non-NFC, control char, surrounding whitespace, contains `</c>`, trailing `.`, > 200 chars, comma where forbidden, > 10 categories, …). `message` names exactly what failed and the reference URL. The connection stays open. |
| `"invalid_image"`   | The JPEG payload is unusable — bad/missing SOI, decode failure, CMYK, EXIF/ICC/colour-profile rejection, out-of-range dimensions, or over the patch budget. |
| `"model_deviation"` | The model emitted geometry but **zero** of it was valid for the task (all off-shape / unlabeled), or the output was unparseable gibberish — nothing usable to return (see the `boxes`/`points`/`abstained` distinction above). |
| `"internal"`        | A server-side fault: an internal IPC-contract violation, or a worker self-exit cause (CUDA OOM / inference timeout — those are followed by a `Close(1011)`; see below). |

### Two error surfaces — and only two

* `type:"error"` JSON message — per-frame failure where the framing is
  still intact (slot-validation rejection → `invalid_request`, JPEG
  malformed / out-of-range → `invalid_image`, model deviation →
  `model_deviation`, server fault → `internal`). For most per-frame
  errors the WebSocket stays open and the next Frame proceeds normally.
  **Two exceptions**, both reported as `code:"internal"`: the inference
  timeout (exceeded `LA_INFERENCE_TIMEOUT_S`, default 600 s; `message`
  starts `"inference timeout:"`) and CUDA OOM (`message` starts
  `"CUDA out of memory"`) are followed *immediately* by a `Close(1011)`
  because the worker self-exits to restore CUDA state from scratch. The
  typed error reaches the client first (the worker awaits the UDS write
  before calling `os._exit`), then the Close. Reconnect after ~10–15 s
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

async def run(frames, request, generation_mode="slow"):
    # `request` is a typed PromptRequest dict, e.g.
    #   {"task": "point", "phrase": "drone in the sky"}
    #   {"task": "detection", "categories": ["drone", "bird"]}
    # You can also lift one verbatim from caps["preset_prompts"].
    caps = fetch_caps()
    assert caps["model"] == "nvidia/LocateAnything-3B"
    async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:
        for fid, jpeg in frames:
            header = json.dumps({
                "frame_id":        fid,
                "request":         request,
                "generation_mode": generation_mode,
                "jpeg_len":        len(jpeg),
            }).encode()
            await ws.send(struct.pack(">I", len(header)) + header + jpeg)
            obj = json.loads(await ws.recv())
            # Strict sum type: match the tag, read exactly that variant's
            # fields. No defensive "is it present?" checks needed.
            match obj["type"]:
                case "boxes":     handle_boxes(obj["boxes"])
                case "points":    handle_points(obj["points"])
                case "abstained": handle_abstained(obj)   # nothing to draw
                case "error":     handle_error(obj["code"], obj["message"])
```

The reference Rust client (using `tokio-tungstenite`) follows the same
pattern. See `examples/reference_client.py` for a full backpressure-
respecting, reconnect-with-backoff implementation.
