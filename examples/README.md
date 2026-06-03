# Examples

Working code that exercises the server's external API. None of these
files are needed to *run* the server — only to demonstrate
correct client behavior.

## `reference_client.py`

Live streaming client. Reads frames from an OpenCV source (webcam,
video file, or RTSP URL), encodes each as JPEG, sends over the
`/v1/stream` WebSocket, and prints each reply by its variant — a
`boxes` / `points` / `abstained` / `error` tagged union (A.2). Honors
backpressure end to end (camera → queue → WebSocket).

The request is a TYPED sum over the seven trained tasks, not a free
prompt string: pick a `--task` and fill that task's slot. The server
compiles the typed request to the exact trained prompt and validates
the slot server-side (a bad slot comes back as `error{code:
"invalid_request"}`). `/v1/capabilities.preset_prompts` advertises
ready-made typed `request` objects to copy.

| `--task` | slot flag | example |
| --- | --- | --- |
| `detection` | `--categories a,b,c` (1..=10) | `--categories person,laptop,cup` |
| `phrase_single` | `--phrase` | `--phrase 'the red car'` |
| `phrase_multi` | `--phrase` | `--phrase 'a small drone in the sky'` |
| `point` | `--phrase` (no comma) | `--phrase 'drone in the sky'` |
| `text_grounding` | `--text` | `--text STOP` |
| `gui_box` | `--description` | `--description 'the search button'` |
| `scene_text` | (none) | |

```bash
# Install the two non-stdlib deps
pip install --user "websockets==13.1" "opencv-python-headless==4.11.0.86"

# Webcam (assuming /dev/video0) — closed-class detection
python reference_client.py --source 0 \
    --task detection --categories person,laptop,cup \
    --mode hybrid

# Video file — point at the drone
python reference_client.py --source path/to/clip.mp4 \
    --task point --phrase 'drone in the sky' \
    --mode slow

# RTSP stream — multi-instance phrase grounding
python reference_client.py --source 'rtsp://192.168.1.10/cam1' \
    --task phrase_multi --phrase 'a small drone in the sky' \
    --mode slow

# Scene text takes no slot
python reference_client.py --source path/to/sign.mp4 \
    --task scene_text --mode hybrid
```

Each reply is exactly one variant — `boxes` (each box has a required
`label` + pixel `bbox_px`), `points` (each has `label` + `point_px`),
`abstained` (the model cleanly found nothing), or `error` (one of the
four codes `invalid_request` / `invalid_image` / `model_deviation` /
`internal`). The client `match`es the `type` tag and reads only that
variant's fields — strict typing means zero downstream defensiveness.

Read the source — it's heavily commented, and the comments document
the *why* of each design choice (backpressure, frame_id correlation,
reconnect semantics).
