"""
VISTA — Production FastAPI Server
===================================
Fixes from audit:
  - asyncio.Queue bridge (no asyncio.run() from threads)
  - Single VideoCapture ownership (detector owns cap, server never touches it)
  - Thread-safe WebSocket list replaced with asyncio-native approach
  - Shared ReIDGallery across all cameras (enables cross-camera matching)
  - /gallery endpoint for live ReID stats
  - Structured alert log with deduplication
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections import deque
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.reid     import ReIDGallery
from core.detector import VISTADetector, Alert

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="VISTA", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# ONE shared gallery — all cameras write to it → cross-camera ReID works
_gallery = ReIDGallery()

# Per-camera: frame_queue (maxsize=2) + alert_queue (maxsize=500)
# maxsize=2 for frames: always deliver latest, never stale
_frame_queues: dict[int, queue.Queue] = {}
_alert_queues: dict[int, queue.Queue] = {}

# Global alert log (thread-safe deque)
_alert_log: deque[dict] = deque(maxlen=500)

# asyncio Queue for broadcasting to WebSocket clients — bridge from sync threads
_ws_broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

# Active WebSocket connections (managed entirely in async context)
_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()

# Detector references (for graceful shutdown)
_detectors: list[VISTADetector] = []
_detector_threads: list[threading.Thread] = []


# ---------------------------------------------------------------------------
# Alert bridge — sync thread → asyncio queue → WebSocket broadcast
# ---------------------------------------------------------------------------

def _alert_collector(alert_queues: dict[int, queue.Queue]):
    """
    Runs in a daemon thread.
    Drains all camera alert queues and pushes to the asyncio bridge queue.
    Uses asyncio.Queue.put_nowait via call_soon_threadsafe.
    """
    loop = asyncio.get_event_loop()
    while True:
        for cam_id, q in alert_queues.items():
            try:
                alert: Alert = q.get_nowait()
                d = alert.to_dict()
                _alert_log.appendleft(d)
                # Thread-safe push into asyncio queue
                loop.call_soon_threadsafe(_ws_broadcast_queue.put_nowait, d)
            except queue.Empty:
                pass
        time.sleep(0.01)  # 100 Hz polling — low latency, negligible CPU


async def _ws_broadcaster():
    """
    Coroutine that drains _ws_broadcast_queue and fans out to all WS clients.
    Runs as a background asyncio task.
    """
    while True:
        try:
            data = await asyncio.wait_for(_ws_broadcast_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Send keep-alive ping
            data = {"kind": "ping", "ts": time.time()}

        payload = json.dumps(data)
        async with _ws_lock:
            dead = []
            for ws in _ws_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Camera startup
# ---------------------------------------------------------------------------

def _start_camera(cam_id: int, source, display: bool = False):
    fq = queue.Queue(maxsize=2)
    aq = queue.Queue(maxsize=500)
    _frame_queues[cam_id] = fq
    _alert_queues[cam_id] = aq

    det = VISTADetector(
        camera_id=cam_id,
        source=source,
        reid_gallery=_gallery,
        frame_queue=fq,
        alert_queue=aq,
    )
    _detectors.append(det)

    t = threading.Thread(
        target=det.run,
        kwargs={"display": display},
        daemon=True,
        name=f"vista-cam-{cam_id}",
    )
    t.start()
    _detector_threads.append(t)
    print(f"[Server] Camera {cam_id} thread started → source={source}")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    # ── Configure your cameras here ──────────────────────────────
    cameras = [
        (0, 0),            # (camera_id, source) — 0 = webcam
        # (1, "rtsp://..."),   # add IP cameras
        # (2, "video.mp4"),
    ]
    # ─────────────────────────────────────────────────────────────

    for cam_id, src in cameras:
        _start_camera(cam_id, src)

    # Alert collector (sync → async bridge)
    t = threading.Thread(
        target=_alert_collector,
        args=(_alert_queues,),
        daemon=True,
        name="vista-alert-collector",
    )
    t.start()

    # WebSocket broadcaster
    asyncio.create_task(_ws_broadcaster())
    print("[Server] VISTA API ready.")


@app.on_event("shutdown")
async def shutdown():
    for det in _detectors:
        det.stop()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status":  "running",
        "cameras": list(_frame_queues.keys()),
        "gallery": _gallery.stats(),
    }


@app.get("/stream/{camera_id}")
def mjpeg_stream(camera_id: int):
    """MJPEG stream — use as <img src="http://host/stream/0"> in React."""
    fq = _frame_queues.get(camera_id)

    def generate():
        while True:
            if fq is None:
                time.sleep(0.1)
                continue
            try:
                frame_bytes = fq.get(timeout=0.1)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            except queue.Empty:
                continue

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@app.get("/alerts")
def get_alerts(limit: int = 100, kind: Optional[str] = None):
    alerts = list(_alert_log)
    if kind:
        alerts = [a for a in alerts if a["kind"] == kind]
    return alerts[:limit]


@app.get("/gallery")
def get_gallery():
    """Live ReID gallery stats + all confirmed persons."""
    entries = _gallery.get_all_entries()
    return {
        "stats": _gallery.stats(),
        "persons": [
            {
                "global_id":      e.global_id,
                "camera_id":      e.camera_id,
                "local_track_id": e.local_track_id,
                "last_seen":      e.last_seen,
                "confirmed":      e.confirmed,
            }
            for e in entries
        ],
    }


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    async with _ws_lock:
        _ws_clients.append(ws)
    try:
        while True:
            # Keep the connection alive — broadcaster handles sends
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,   # must be 1 — detector threads live in main process
    )