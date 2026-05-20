"""
VISTA — VideoMAE Fall Detector
================================
Model: yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect
Classes: FallDown, LyingDown, SitDown, Sitting, StandUp, Standing, Walking
Runs async in background thread — never blocks main pipeline.

Integration:
    detector = FallDetector()
    # Feed 16 frames per track, get fall probability back
    detector.submit(track_id, frames_buffer)
    result = detector.get_result(track_id)  # None if not ready
"""

import threading
import queue
import time
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


FALL_CLASSES    = {0, 1}   # FallDown, LyingDown
NUM_FRAMES      = 16       # VideoMAE expects exactly 16 frames
FALL_THRESHOLD  = 0.60     # confidence to trigger alert
MODEL_ID        = "yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect"


class FallDetector:
    """
    Async VideoMAE fall detector.
    Maintains a 16-frame rolling buffer per track.
    Processes clips in background — zero impact on main FPS.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FallDetector] Loading VideoMAE from {MODEL_ID}...")
        self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        self.model     = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()
        print(f"[FallDetector] Ready on {self.device}")

        # Per-track frame buffer: track_id → deque of RGB frames (H,W,3)
        self._buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=NUM_FRAMES))
        # Results: track_id → (fall_prob, timestamp)
        self._results: dict[int, tuple] = {}
        self._display_ids: dict[int, str] = {}
        self._lock = threading.Lock()

        # Work queue: (track_id, list_of_frames)
        self._queue  = queue.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def push_frame(self, track_id: int, crop_bgr: np.ndarray, display_id: str = None):
        """
        Push a person crop (BGR) into the track's frame buffer.
        Call every frame for each tracked person.
        When buffer hits 16 frames, submit for inference automatically.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return
        # Resize to 224×224 for VideoMAE
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        with self._lock:
            if display_id:
                self._display_ids[track_id] = display_id
            self._buffers[track_id].append(rgb)
            buf = self._buffers[track_id]
            if len(buf) == NUM_FRAMES:
                frames = list(buf)
                buf.clear()   # reset buffer after submission
                try:
                    self._queue.put_nowait((track_id, frames))
                except queue.Full:
                    pass   # drop if worker is busy

    def get_result(self, track_id: int) -> Optional[tuple]:
        """
        Returns (fall_prob, timestamp) if a result is ready, else None.
        Result is consumed once read.
        """
        with self._lock:
            return self._results.pop(track_id, None)

    def purge(self, active_ids: set):
        """Remove buffers for tracks that no longer exist."""
        with self._lock:
            dead = [k for k in self._buffers if k not in active_ids]
            for k in dead:
                del self._buffers[k]
                self._display_ids.pop(k, None)

    @torch.no_grad()
    def _worker(self):
        while True:
            try:
                track_id, frames = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                inputs = self.processor(frames, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                if self.device.type == "cuda":
                    pixel_values = pixel_values.half()

                logits = self.model(pixel_values).logits
                probs  = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()

                # Fall probability = FallDown + LyingDown
                fall_prob = float(probs[0] + probs[1])
                pred_class = int(probs.argmax())
                pred_label = self.model.config.id2label[pred_class]

                with self._lock:
                    display_id = self._display_ids.get(track_id, f"tid={track_id}")

                print(f"[FallDetector] {display_id} → {pred_label} "
                      f"(fall_prob={fall_prob:.2f})")

                with self._lock:
                    self._results[track_id] = (fall_prob, time.time())

            except Exception as e:
                print(f"[FallDetector] error: {e}")
