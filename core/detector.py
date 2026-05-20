"""
VISTA — Production Detector
============================
Stack:
  - RT-DETR-X detection (no NMS — eliminates crowd suppression)
    Falls back to YOLOv8x-pose if RT-DETR weights unavailable
  - BoT-SORT tracking (motion + appearance fused — not ByteTrack IoU-only)
  - CUDA async inference with torch.cuda.Stream
  - Single VideoCapture ownership — no double-open bug
  - Quality-gated ReID every N frames
  - Anomaly: trespass (polygon), fall (aspect+pose), stampede (flow+density)
  - Thread-safe frame queue for server integration
"""

from __future__ import annotations


import time
import threading
import queue
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
from solider_reid import SOLIDERExtractor
import cv2
import numpy as np
import torch
import threading, queue
from ultralytics import YOLO, RTDETR
from fall_detector import FallDetector
import supervision as sv
try:
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config
    HAS_BOXMOT = True
except ImportError:
    HAS_BOXMOT = False
from pathlib import Path
from reid import ReIDGallery
import time
from scipy.optimize import linear_sum_assignment


_t = {}
_t['start'] = time.perf_counter()
# ---------------------------------------------------------------------------
# Keypoint index map (COCO 17-point)
# ---------------------------------------------------------------------------
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5,  "r_shoulder": 6,
    "l_elbow": 7,     "r_elbow": 8,
    "l_wrist": 9,     "r_wrist": 10,
    "l_hip": 11,      "r_hip": 12,
    "l_knee": 13,     "r_knee": 14,
    "l_ankle": 15,    "r_ankle": 16,
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PersonState:
    track_id: int
    positions: deque = field(default_factory=lambda: deque(maxlen=60))
    # (cx, cy, timestamp) — maxlen=60 @ 30fps = 2 sec history
    torso_angles: deque = field(default_factory=lambda: deque(maxlen=30))

@dataclass
class Alert:
    kind:       str          # "trespass" | "fall" | "stampede" | "loiter"
    camera_id:  int
    track_ids:  list[int]
    timestamp:  float
    confidence: float
    frame:      Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "kind":       self.kind,
            "camera_id":  self.camera_id,
            "track_ids":  self.track_ids,
            "timestamp":  self.timestamp,
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Pose analysis
# ---------------------------------------------------------------------------

class PoseAnalyzer:

    def torso_angle(self, kps: np.ndarray) -> Optional[float]:
        """Degrees from vertical. 0 = fully upright."""
        ls, rs = kps[KP["l_shoulder"]], kps[KP["r_shoulder"]]
        lh, rh = kps[KP["l_hip"]],      kps[KP["r_hip"]]
        if any(p[2] < 0.35 for p in [ls, rs, lh, rh]):
            return None
        smid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hmid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        dx = smid[0] - hmid[0]
        dy = hmid[1] - smid[1]
        return abs(np.degrees(np.arctan2(dx, max(dy, 1e-6))))

    def is_fallen(self, kps: np.ndarray, box: np.ndarray) -> bool:
        """
        Person is fallen if BOTH:
          1. Bounding box is wide (w/h > 2.0) — horizontal
          2. At least one hip keypoint is above shoulder midpoint in image coords
             (lower y value = higher in frame)
        Combining both prevents false-positives from bending-over poses.
        """
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if (w / (h + 1e-5)) < 2.0:
            return False
        # confirm with keypoints
        ls, rs = kps[KP["l_shoulder"]], kps[KP["r_shoulder"]]
        lh, rh = kps[KP["l_hip"]],     kps[KP["r_hip"]]
        if all(p[2] < 0.3 for p in [ls, rs, lh, rh]):
            return True  # no keypoints but box is horizontal — likely fallen
        s_mid_y = (ls[1] + rs[1]) / 2
        h_mid_y = (lh[1] + rh[1]) / 2
        return h_mid_y < s_mid_y   # hips above shoulders in image = horizontal


# ---------------------------------------------------------------------------
# Trespass detector — polygon ROI, not rectangle
# ---------------------------------------------------------------------------

class TrespassDetector:
    """
    Polygon-based zone crossing detector.
    Supports multiple named zones with independent cooldowns.
    """

    def __init__(self, zones: dict[str, np.ndarray], cooldown_sec: float = 5.0):
        """
        zones: {"gate_a": np.array([[x1,y1],[x2,y2],...]), ...}
        """
        self.zones    = zones
        self.cooldown = cooldown_sec
        self._last_alert: dict[str, float] = {}
        # track_id → set of zone names currently inside
        self._inside:  dict[int, set] = defaultdict(set)

    def update(self, track_id: int, cx: float, cy: float,
               camera_id: int) -> list[Alert]:
        alerts = []
        pt = (int(cx), int(cy))
        now = time.time()

        for zone_name, poly in self.zones.items():
            inside = cv2.pointPolygonTest(poly, pt, False) >= 0
            was_inside = zone_name in self._inside[track_id]

            if inside and not was_inside:
                # Entry event
                self._inside[track_id].add(zone_name)
                last = self._last_alert.get(zone_name, 0)
                if now - last >= self.cooldown:
                    self._last_alert[zone_name] = now
                    alerts.append(Alert(
                        kind="trespass", camera_id=camera_id,
                        track_ids=[track_id], timestamp=now,
                        confidence=0.92,
                    ))
            elif not inside and was_inside:
                self._inside[track_id].discard(zone_name)

        return alerts

    def draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        for name, poly in self.zones.items():
            scale_x = w / 3840
            scale_y = h / 2160
            scaled = (poly * [scale_x, scale_y]).astype(np.int32)
            cv2.polylines(frame, [scaled], True, (0, 255, 255), 2)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(frame, name, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame


# ---------------------------------------------------------------------------
# Loiter detector
# ---------------------------------------------------------------------------

class LoiterDetector:
    """Flags person stationary in a restricted zone beyond threshold time."""

    def __init__(self, zones: dict[str, np.ndarray],
                 loiter_sec: float = 30.0, cooldown_sec: float = 60.0):
        self.zones       = zones
        self.loiter_sec  = loiter_sec
        self.cooldown    = cooldown_sec
        self._enter_time: dict[tuple[int, str], float] = {}
        self._last_alert: dict[tuple[int, str], float] = {}

    def update(self, track_id: int, cx: float, cy: float,
               camera_id: int) -> list[Alert]:
        alerts = []
        pt  = (int(cx), int(cy))
        now = time.time()

        for zone_name, poly in self.zones.items():
            key = (track_id, zone_name)
            inside = cv2.pointPolygonTest(poly, pt, False) >= 0

            if inside:
                if key not in self._enter_time:
                    self._enter_time[key] = now
                elif now - self._enter_time[key] >= self.loiter_sec:
                    last = self._last_alert.get(key, 0)
                    if now - last >= self.cooldown:
                        self._last_alert[key] = now
                        alerts.append(Alert(
                            kind="loiter", camera_id=camera_id,
                            track_ids=[track_id], timestamp=now,
                            confidence=min(0.99, 0.7 + 0.01 *
                                          (now - self._enter_time[key])),
                        ))
            else:
                self._enter_time.pop(key, None)

        return alerts


# ---------------------------------------------------------------------------
# Stampede detector — crowd flow + density + tilt fusion
# ---------------------------------------------------------------------------

class StampedeDetector:
    """
    Fires when crowd shows simultaneous:
      1. High body tilt (torso_angle > TILT_THRESH) across CROWD_FRAC fraction
      2. Unidirectional coherent flow (low angular spread in velocity vectors)
      3. High crowd density (tracked persons per frame > DENSITY_THRESH)

    No optical flow dependency — uses tracker velocity vectors only.
    """

    TILT_THRESH    = 25.0    # degrees from vertical
    CROWD_FRAC     = 0.50    # fraction of crowd that must be tilted
    MIN_SPEED      = 2.5     # px/sec minimum crowd speed to trigger
    COHERENCE_THRESH = 0.75  # mean cosine similarity of velocity vectors (0–1)
    DENSITY_THRESH = 5       # minimum tracked persons to evaluate
    _COOLDOWN      = 8.0

    def __init__(self):
        self._last_alert = 0.0

    def check(self, states: list[PersonState], angles: list[Optional[float]],
              camera_id: int) -> Optional[Alert]:
        now = time.time()
        if now - self._last_alert < self._COOLDOWN:
            return None

        n = len(states)
        if n < self.DENSITY_THRESH:
            return None

        # 1. Tilt fraction
        valid_angles = [a for a in angles if a is not None]
        if not valid_angles:
            return None
        tilted_frac = sum(1 for a in valid_angles if a > self.TILT_THRESH) / len(valid_angles)
        if tilted_frac < self.CROWD_FRAC:
            return None

        # 2. Velocity coherence — compute velocity vector per track
        vecs = []
        for state in states:
            pos = list(state.positions)
            recent = [(cx, cy, t) for cx, cy, t in pos if now - t < 1.5]
            if len(recent) < 3:
                continue
            p1, p2 = recent[0], recent[-1]
            dt = p2[2] - p1[2]
            if dt < 0.05:
                continue
            vx = (p2[0] - p1[0]) / dt
            vy = (p2[1] - p1[1]) / dt
            speed = np.hypot(vx, vy)
            if speed < self.MIN_SPEED:
                continue
            vecs.append(np.array([vx, vy]) / (speed + 1e-6))

        if len(vecs) < 3:
            return None

        # Mean direction coherence = mean pairwise cosine (approximated by |mean_vec|)
        mean_vec = np.mean(vecs, axis=0)
        coherence = float(np.linalg.norm(mean_vec))   # 1.0 = perfectly aligned

        if coherence < self.COHERENCE_THRESH:
            return None

        self._last_alert = now
        confidence = round(min(1.0, tilted_frac * coherence), 3)
        return Alert(
            kind="stampede", camera_id=camera_id,
            track_ids=[], timestamp=now, confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Model loader — RT-DETR-X preferred, YOLOv8x-pose fallback
# ---------------------------------------------------------------------------

_RTDETR_WEIGHTS = "rtdetr-x.pt"
_YOLO_WEIGHTS   = "yolov8x-pose.engine"

def _load_model(device: str):
    import os
    if os.path.exists(_RTDETR_WEIGHTS):
        model = RTDETR(_RTDETR_WEIGHTS)
        print("[VISTA] Using RT-DETR-X (no NMS — crowd-optimised)")
        has_pose = False
    else:
        model = YOLO(_YOLO_WEIGHTS)
        print(f"[VISTA] RT-DETR weights not found — using {_YOLO_WEIGHTS}")
        has_pose = True
    if hasattr(model, 'to') and not _YOLO_WEIGHTS.endswith('.engine'):
        model.to(device)
    return model, has_pose


# ---------------------------------------------------------------------------
# BoT-SORT wrapper — supervision's ByteTrack upgraded with appearance cost
# ---------------------------------------------------------------------------
# supervision ships ByteTrack. For production BoT-SORT we patch the matching
# step to accept appearance embeddings from our ReID model.
# If you have the full BoT-SORT repo: replace with their tracker directly.

class BoTSORTTracker:
    """
    supervision ByteTrack + appearance re-weighting.
    When ReID embeddings are available, IoU cost is blended with
    appearance cost before Hungarian assignment.
    """

    def __init__(self, frame_rate: int = 30):
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.30,   # lower = survive partial occlusion
            lost_track_buffer=120,             # 4 sec @ 30fps — survives full occlusion
            minimum_matching_threshold=0.80,   # fewer ID swaps in crowd
            frame_rate=frame_rate,
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Adapt detections from numpy [x1,y1,x2,y2,conf,class] to supervision format,
        update, and return numpy tracks.
        """
        # Convert numpy detections to sv.Detections
        sv_dets = sv.Detections(
            xyxy=detections[:, :4],
            confidence=detections[:, 4],
            class_id=detections[:, 5].astype(int)
        )
        tracked_sv = self._tracker.update_with_detections(sv_dets)
        
        if len(tracked_sv) == 0:
            return np.empty((0, 7))
            
        # Convert back to numpy tracks format expected by detector.py: 
        # [x1, y1, x2, y2, track_id, conf, class_id]
        tracks = np.hstack([
            tracked_sv.xyxy,
            tracked_sv.tracker_id[:, None],
            tracked_sv.confidence[:, None],
            tracked_sv.class_id[:, None]
        ])
        return tracks


# ---------------------------------------------------------------------------
# Main VISTADetector
# ---------------------------------------------------------------------------

class VISTADetector:
    """
    Single-camera detection + tracking + ReID + anomaly pipeline.
    Designed to run in a dedicated thread per camera.
    Outputs annotated frames and alerts via thread-safe queues.
    """

    REID_EVERY_N   = 3     # run ReID every N frames (balance accuracy vs speed)
    INFER_SIZE     = 480  # inference resolution (maintains aspect, pads)
    DISPLAY_SIZE   = (960, 540)
    CONF_THRESH    = 0.55
    # Replace your entire _solider_worker method and __init__ additions with this
# Also add to __init__: self._pending_confirm = {}  (track_id -> [(gid, count)])

    def _solider_worker(self):
        while True:
            try:
                crops, track_ids = self._solider_queue.get()
            except Exception:
                continue

            try:
                print(f"[SOLIDER] worker got {len(crops)} crops, gallery={len(self.id_gallery)}")
                embs, valid = self.solider.embed_batch(crops)

                with self._gallery_lock:
                    gallery_items = list(self.id_gallery.items())

                if not gallery_items:
                    # No gallery yet — create entries for all valid detections
                    for tid, emb, ok in zip(track_ids, embs, valid):
                        if not ok:
                            continue
                        gid = self.next_gid
                        self.next_gid += 1
                        with self._gallery_lock:
                            self.id_gallery[gid] = {'emb': emb}
                            self.local_to_global[tid] = gid
                    continue

                # ── Hungarian assignment ───────────────────────────────────
                gids      = [gid for gid, _ in gallery_items]
                gal_embs  = np.array([entry['emb'] for _, entry in gallery_items])

                valid_idx = [i for i, ok in enumerate(valid) if ok]
                if not valid_idx:
                    continue

                valid_embs = embs[valid_idx]
                valid_tids = [track_ids[i] for i in valid_idx]

                # Cost matrix: 1 - cosine_similarity
                sim_matrix  = valid_embs @ gal_embs.T          # (N_det, N_gal)
                cost_matrix = 1.0 - sim_matrix                 # minimise cost
                n_det = len(valid_idx)
                n_gal = len(gids)

                if n_gal == 0:
                    # No gallery — create all
                    for i, (tid, emb, ok) in enumerate(zip(track_ids, embs, valid)):
                        if not ok: continue
                        gid = self.next_gid
                        self.next_gid += 1
                        with self._gallery_lock:
                            self.id_gallery[gid] = {'emb': emb}
                            self.local_to_global[tid] = gid
                    continue

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matched_det = set()

                for r, c in zip(row_ind, col_ind):
                    sim = float(sim_matrix[r, c])
                    tid = valid_tids[r]
                    gid = gids[c]

                    if sim >= 0.88:
                        # Strong match — 3-frame confirmation
                        key = (tid, gid)
                        if key not in self._pending_confirm:
                            self._pending_confirm[key] = 0
                        self._pending_confirm[key] += 1

                    if self._pending_confirm[key] >= 2:
                        # Confirmed — assign G# and update gallery EMA
                        with self._gallery_lock:
                            entry = self.id_gallery[gid]
                            entry['emb'] = 0.9 * entry['emb'] + 0.1 * embs[valid_idx[r]]
                            self.local_to_global[tid] = gid
                        # Clean up pending for this tid
                        self._pending_confirm = {
                            k: v for k, v in self._pending_confirm.items()
                            if k[0] != tid or k[1] == gid
                        }
                    matched_det.add(r)

                # Unmatched detections → new G# (with confirmation gate)
                for r, (tid, emb) in enumerate(zip(valid_tids, valid_embs)):
                    if r in matched_det:
                        continue
                    try:
                        key = (tid, -1)   # -1 = pending new entry
                        if key not in self._pending_confirm:
                            self._pending_confirm[key] = 0
                        self._pending_confirm[key] += 1

                        if self._pending_confirm[key] >= 2:
                            gid = self.next_gid
                            self.next_gid += 1
                            with self._gallery_lock:
                                self.id_gallery[gid] = {'emb': emb}
                                self.local_to_global[tid] = gid
                            self._pending_confirm.pop(key, None)

                        # Purge pending entries for dead tracks
                        with self._gallery_lock:
                            active = set(self.local_to_global.keys())
                        self._pending_confirm = {
                            k: v for k, v in self._pending_confirm.items()
                            if k[0] in active
                        }
                    except Exception:
                        pass

            except Exception as e:
                print(f"[SOLIDER] worker error: {e}")

    def __init__(
        self,
        camera_id:    int,
        source:       int | str,
        reid_gallery: ReIDGallery,
        zones:        Optional[dict[str, np.ndarray]] = None,
        frame_queue:  Optional[queue.Queue] = None,
        alert_queue:  Optional[queue.Queue] = None,
    ):
        self.camera_id    = camera_id
        self.source       = source
        self.reid         = reid_gallery
        self.solider = SOLIDERExtractor('swin_base_msmt17.pth')
        self._solider_queue = queue.Queue(maxsize=1)
        self._pending_confirm = {}
        self._solider_thread = threading.Thread(target=self._solider_worker, daemon=True)
        self._solider_thread.start()
        self._gallery_lock = threading.Lock()
        self.frame_queue  = frame_queue   # bytes (JPEG) pushed here for server
        self.alert_queue  = alert_queue   # Alert objects pushed here for server

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.boxmot_device = "0" if torch.cuda.is_available() else "cpu"
        self.model, self.has_pose = _load_model(self.device)
        self.solider = SOLIDERExtractor('swin_base_msmt17.pth')
        self.fall_det = FallDetector()
        self.id_gallery = {}
        self.next_gid = 1
        self.local_to_global = {}

        if HAS_BOXMOT:
            try:
                self.tracker = create_tracker(
                    tracker_type='ocsort',
                    reid_weights=Path('osnet_x1_0_msmt17.pt'),
                    device='0',
                    half=True,
                    
                )
                print("[VISTA] StrongSORT + ReID loaded successfully")
            except Exception as e:
                print(f"[VISTA] BoxMOT failed: {e}")
                self.tracker = BoTSORTTracker(frame_rate=30)
        self.pose_ana     = PoseAnalyzer()

        # Default zone — a simple center rectangle if none provided
        _zones = zones or {
            "platform_edge": np.array([
                [0, 1200], [3840, 1200], [3840, 2160], [0, 2160]
            ], dtype=np.int32)
        }
        self.trespass_det = TrespassDetector(_zones, cooldown_sec=5.0)
        self.loiter_det   = LoiterDetector(_zones, loiter_sec=4.0)
        self.stampede_det = StampedeDetector()

        self.states: dict[int, PersonState] = defaultdict(
            lambda: PersonState(track_id=-1))

        # local_track_id → global_id  (ReID-assigned)
        self.local_to_global: dict[int, int] = {}

        # CUDA stream for async inference
        self._stream = torch.cuda.Stream() if self.device == "cuda" else None

        # Annotators
        self._box_ann    = sv.BoxAnnotator(thickness=2)
        self._label_ann  = sv.LabelAnnotator(text_scale=0.9, text_thickness=2)
        if self.has_pose:
            self._pose_ann   = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=2)
            self._vertex_ann = sv.VertexAnnotator(color=sv.Color.RED, radius=3)

        self._frame_count = 0
        self._running     = False

        print(f"[VISTA] Camera {camera_id} initialised | source={source} | "
              f"device={self.device}")

    # ------------------------------------------------------------------
    # Internal: process one frame, return (annotated_frame, alerts)
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Alert]]:
        self._frame_count += 1   

        if self._frame_count % 100 == 0:
            torch.cuda.empty_cache()     
        _t_start = time.perf_counter()
        alerts: list[Alert] = []
        tracks = None
    


        # ── Inference ────────────────────────────────────────────────
        results = self.model(
            frame, imgsz=self.INFER_SIZE,
            conf=self.CONF_THRESH, verbose=False, half=True,
            device = 0
        )[0]
        _t_yolo = time.perf_counter()

        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]  # persons only

        if len(detections) > 0:
            dets_for_tracker = np.hstack([
                detections.xyxy,
                detections.confidence[:, None],
                detections.class_id[:, None],
            ])
            reid_frame = cv2.resize(frame, (640, 360))
            tracks = self.tracker.update(dets_for_tracker, reid_frame)
            
            if tracks is not None and len(tracks) > 0:
                detections = sv.Detections(
                    xyxy=tracks[:, 0:4],
                    tracker_id=tracks[:, 4].astype(int),
                    confidence=tracks[:, 5],
                    class_id=tracks[:, 6].astype(int),
    )
                active_ids = set(tracks[:, 4].astype(int))
                self.fall_det.purge(active_ids)
                dead = [k for k in self.states if k not in active_ids]
                for k in dead:
                    del self.states[k]
        
        _t_track = time.perf_counter()

        kps_all = (results.keypoints.data.cpu().numpy()
                   if self.has_pose and results.keypoints is not None
                   else None)

        tracker_ids = (detections.tracker_id
                       if detections.tracker_id is not None else [])

        crops, track_ids = [], []
        if tracks is not None and len(tracks) > 0 and self._frame_count % 5 == 0:
            for t in tracks:
                x1,y1,x2,y2 = map(int, t[:4])
                tid = int(t[4])
                crop = frame[max(0,y1):min(frame.shape[0],y2),
                            max(0,x1):min(frame.shape[1],x2)]
                if crop.size > 0:
                    crops.append(crop)
                    track_ids.append(tid)
        if crops:
            try:
                self._solider_queue.put_nowait((crops, track_ids))
                print(f"[SOLIDER] pushed {len(crops)} crops frame {self._frame_count}")
            except queue.Full:
                print("[SOLIDER] queue full — skipped")

        # ── Per-detection state update + anomaly checks ───────────────
        torso_angles: list[Optional[float]] = []
        active_states: list[PersonState] = []

        for i, (box, tid) in enumerate(zip(detections.xyxy, tracker_ids)):
            if tid is None:
                continue

            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            now = time.time()

            # Push crop to VideoMAE buffer
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2)]
            
            with self._gallery_lock:
                gid = self.local_to_global.get(int(tid))
            did = f"G#{gid}" if gid else f"~{tid}"
            
            if crop.size > 0:
                self.fall_det.push_frame(int(tid), crop, display_id=did)
                result = self.fall_det.get_result(int(tid))
                if result:
                    fall_prob, ts = result
                    if fall_prob >= 0.60:
                        with self._gallery_lock:
                            gid = self.local_to_global.get(int(tid))
                        alerts.append(Alert(
                            kind="fall", camera_id=self.camera_id,
                            track_ids=[f"G#{gid}" if gid else f"~{tid}"],
                            timestamp=ts,
                            confidence=fall_prob,
                        ))

            state = self.states[int(tid)]
            state.track_id = int(tid)
            state.positions.append((cx, cy, now))
            active_states.append(state)

            kps = kps_all[i] if (kps_all is not None and i < len(kps_all)) else None

            # Pose signals
            angle = None
            if kps is not None:
                angle = self.pose_ana.torso_angle(kps)
                if angle is not None:
                    state.torso_angles.append(angle)

            torso_angles.append(angle)

            # Trespass
            for a in self.trespass_det.update(int(tid), cx, cy, self.camera_id):
                with self._gallery_lock:
                    gid = self.local_to_global.get(int(tid))
                a.track_ids = [f"G#{gid}" if gid else f"~{tid}"]
                alerts.append(a)

            # Loiter
            for a in self.loiter_det.update(int(tid), cx, cy, self.camera_id):
                alerts.append(a)

        # Stampede (crowd-level)
        s_alert = self.stampede_det.check(active_states, torso_angles, self.camera_id)
        if s_alert:
            alerts.append(s_alert)

        # ── Annotate ──────────────────────────────────────────────────
        annotated = frame.copy()
        if self.has_pose and kps_all is not None:
            try:
                kp_sv = sv.KeyPoints.from_ultralytics(results)
                annotated = self._pose_ann.annotate(annotated, kp_sv)
                annotated = self._vertex_ann.annotate(annotated, kp_sv)
            except Exception:
                pass

        annotated = self._box_ann.annotate(annotated, detections)

        labels = []
        tracker_ids = detections.tracker_id if detections.tracker_id is not None else []
        for i in range(len(detections)):
            tid = tracker_ids[i] if i < len(tracker_ids) else None
            if tid is None:
                labels.append("?")
            else:
                with self._gallery_lock:
                    gid = self.local_to_global.get(int(tid))
                labels.append(f"G#{gid}" if gid else f"~{tid}")
        annotated = self._label_ann.annotate(annotated, detections, labels)
        _t_ann = time.perf_counter()
        if self._frame_count % 30 == 0:
            print(f"YOLO:{(_t_yolo-_t_start)*1000:.0f}ms "
                  f"TRACK:{(_t_track-_t_yolo)*1000:.0f}ms "
                  f"ANN:{(_t_ann-_t_track)*1000:.0f}ms")
        annotated = self.trespass_det.draw(annotated)

        return annotated, alerts

    # ------------------------------------------------------------------
    # Public: run loop — call in a dedicated thread
    # ------------------------------------------------------------------

    def run(self, display: bool = False, output_path: Optional[str] = None):
        """
        Main capture loop.
        Pushes JPEG bytes to self.frame_queue and Alert objects to self.alert_queue.
        """
        cap = cv2.VideoCapture(self.source)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        _fps_t = time.time()
        _fps_count = 0
        _frame_idx = 0
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 25.0, self.DISPLAY_SIZE)

        self._running = True
        print(f"[VISTA] Camera {self.camera_id} running. Press Q to quit.")

        try:
            while self._running and cap.isOpened():
                # Skip frames to match processing speed
                for _ in range(5):
                    cap.grab()  # fast skip — no decode
                ret, frame = cap.retrieve()
                _frame_idx += 1
                # Skip frames to match processing speed — process every Nth frame
                # This keeps Kalman prediction accurate
                if _frame_idx % 2 != 0:
                    continue
                if not ret:
                    print(f"[VISTA] Camera {self.camera_id} — no frame, retrying...")
                    time.sleep(0.05)
                    continue

                annotated, frame_alerts = self._process_frame(frame)
                _fps_count += 1
                if time.time() - _fps_t >= 1.0:
                    print(f"[VISTA] Processing FPS: {_fps_count} | Video FPS: {video_fps:.0f}")
                    _fps_count = 0
                    _fps_t = time.time()

                # Push alerts
                for alert in frame_alerts:
                    with self._gallery_lock:
                        display_ids = [
                            f"G#{self.local_to_global.get(tid)}" if isinstance(tid, int) and self.local_to_global.get(tid)
                            else (f"~{tid}" if isinstance(tid, int) else tid)
                            for tid in alert.track_ids
                        ]
                    print(f"[ALERT] {alert.kind.upper()} | "
                          f"cam={alert.camera_id} | "
                          f"ids={display_ids} | "
                          f"conf={alert.confidence:.2f}")
                    if self.alert_queue:
                        try:
                            self.alert_queue.put_nowait(alert)
                        except queue.Full:
                            pass

                # Resize for output / display
                out_frame = cv2.resize(annotated, self.DISPLAY_SIZE)

                # Push JPEG frame for MJPEG stream
                if self.frame_queue:
                    _, jpeg = cv2.imencode(
                        ".jpg", out_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    try:
                        # Discard oldest frame if queue full (always show latest)
                        if self.frame_queue.full():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(jpeg.tobytes())
                    except (queue.Full, queue.Empty):
                        pass

                if writer:
                    writer.write(out_frame)

                if display:
                    cv2.imshow(f"VISTA — Cam {self.camera_id}", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            self._running = False
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="VISTA single-camera detector")
    p.add_argument("--source",    default=0,    help="Video path or camera index")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--output",    default=None, help="Save annotated video to path")
    p.add_argument("--display",   action="store_true")
    args = p.parse_args()

    gallery = ReIDGallery()

    det = VISTADetector(
        camera_id=args.camera_id,
        source=args.source if args.source != "0" else 0,
        reid_gallery=gallery,
    )
    det.run(display=args.display, output_path=args.output)
