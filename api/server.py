"""
VISTA — FastAPI Backend
Streams annotated MJPEG + exposes alert WebSocket + REST logs
"""

import asyncio
import cv2
import json
import time
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.detector import VISTADetector, Alert

app = FastAPI(title="VISTA API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- Shared state ----------
alert_log: deque[dict] = deque(maxlen=200)
latest_frame: dict[int, bytes] = {}          # camera_id → JPEG bytes
active_ws: list[WebSocket] = []
detector_threads: list[threading.Thread] = []


def alert_to_dict(a: Alert) -> dict:
    return {
        "kind": a.kind,
        "camera_id": a.camera_id,
        "track_ids": a.track_ids,
        "timestamp": a.timestamp,
        "confidence": a.confidence,
    }


def run_detector(camera_id: int, source):
    """Runs detector in a background thread, feeds frames + alerts to shared state."""
    det = VISTADetector(camera_id=camera_id, source=source)
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, frame_alerts = det._process_frame(frame)

        # Encode frame as JPEG for MJPEG stream
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        latest_frame[camera_id] = jpeg.tobytes()

        for alert in frame_alerts:
            d = alert_to_dict(alert)
            alert_log.appendleft(d)
            # Broadcast to all connected WebSocket clients
            asyncio.run(_broadcast(d))

    cap.release()


async def _broadcast(data: dict):
    dead = []
    for ws in active_ws:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        active_ws.remove(ws)


# ---------- Routes ----------

@app.get("/")
def root():
    return {"status": "VISTA running", "cameras": list(latest_frame.keys())}


@app.get("/stream/{camera_id}")
def mjpeg_stream(camera_id: int):
    """MJPEG stream for React <img> tag."""
    def generate():
        while True:
            frame = latest_frame.get(camera_id)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)   # ~30 FPS cap

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/alerts")
def get_alerts(limit: int = 50):
    return list(alert_log)[:limit]


@app.get("/alerts/{kind}")
def get_alerts_by_kind(kind: str, limit: int = 20):
    return [a for a in alert_log if a["kind"] == kind][:limit]


@app.websocket("/ws/alerts")
async def alert_websocket(ws: WebSocket):
    await ws.accept()
    active_ws.append(ws)
    try:
        while True:
            await asyncio.sleep(30)   # keep-alive ping
            await ws.send_text(json.dumps({"kind": "ping"}))
    except WebSocketDisconnect:
        active_ws.remove(ws)


@app.on_event("startup")
def startup():
    """Start detector threads for each camera source."""
    sources = [
        (0, 0),            # (camera_id, source) — 0 = webcam
        # (1, "path/to/video.mp4"),   # add more cameras here
    ]
    for cam_id, src in sources:
        t = threading.Thread(target=run_detector, args=(cam_id, src), daemon=True)
        t.start()
        detector_threads.append(t)
        print(f"[VISTA] Detector started: camera {cam_id} → {src}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
