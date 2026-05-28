# Examples

Working code that exercises the server's external API. None of these
files are needed to *run* the server — only to demonstrate
correct client behavior.

## `reference_client.py`

Live streaming client. Reads frames from an OpenCV source (webcam,
video file, or RTSP URL), encodes each as JPEG, sends over the
`/v1/stream` WebSocket, prints detections. Honors backpressure end
to end (camera → queue → WebSocket).

```bash
# Install the two non-stdlib deps
pip install --user "websockets==13.1" "opencv-python-headless==4.11.0.86"

# Webcam (assuming /dev/video0)
python reference_client.py --source 0 \
    --prompt 'Locate all the instances that matches the following description: person</c>laptop</c>cup.' \
    --mode hybrid

# Video file
python reference_client.py --source path/to/clip.mp4 \
    --prompt 'Point to: drone in the sky.' \
    --mode slow

# RTSP stream
python reference_client.py --source 'rtsp://192.168.1.10/cam1' \
    --prompt 'Locate all the instances that match the following description: a small drone in the sky.' \
    --mode slow
```

Read the source — it's heavily commented, and the comments document
the *why* of each design choice (backpressure, frame_id correlation,
reconnect semantics).
