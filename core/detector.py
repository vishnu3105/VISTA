"""
VISTA — Core Detector
YOLOv8n-pose + ByteTrack tracking + pose-based alert logic
"""

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from reid import ReIDGallery
import time

# ---------- Keypoint indices (COCO 17-point) ----------
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

@dataclass
class PersonState:
    track_id: int
    positions: list = field(default_factory=list)   # (cx, cy, timestamp)
    torso_angles: list = field(default_factory=list)
    keypoints_history: list = field(default_factory=list)

@dataclass
class Alert:
    kind: str           # "ticket_abuse" | "stampede" | "fall" | "push"
    camera_id: int
    track_ids: list
    timestamp: float
    confidence: float
    frame: Optional[np.ndarray] = None


class PoseAnalyzer:
    """Derives biomechanical signals from 17 keypoints."""

    def torso_angle(self, kps: np.ndarray) -> Optional[float]:
        """Angle of torso relative to vertical (degrees). 0 = upright."""
        ls, rs = kps[KP["l_shoulder"]], kps[KP["r_shoulder"]]
        lh, rh = kps[KP["l_hip"]],      kps[KP["r_hip"]]
        if any(p[2] < 0.3 for p in [ls, rs, lh, rh]):
            return None
        shoulder_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        hip_mid      = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        dx = shoulder_mid[0] - hip_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]          # y flipped in image coords
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        return angle

    def arms_raised(self, kps: np.ndarray) -> bool:
        """Both wrists above shoulders — panic/fall signal."""
        lw, rw = kps[KP["l_wrist"]], kps[KP["r_wrist"]]
        ls, rs = kps[KP["l_shoulder"]], kps[KP["r_shoulder"]]
        if any(p[2] < 0.3 for p in [lw, rw, ls, rs]):
            return False
        return lw[1] < ls[1] and rw[1] < rs[1]    # lower y = higher in frame

    def is_fallen(self, kps: np.ndarray, box: np.ndarray) -> bool:
        """Person is horizontal if bbox width >> height."""
        x1, y1, x2, y2 = box
        w, h = x2-x1, y2-y1
        return (w / (h + 1e-5)) > 2.5            # aspect ratio threshold


class TicketAbuseDetector:
    """
    Flags when multiple track IDs pass through an entry ROI
    within a single scan window.
    """


    def __init__(self, roi: tuple, window_sec: float = 2.0):
        # roi = (x1, y1, x2, y2) in pixel coords
        self.roi = roi
        self.window = window_sec
        self.entries: list = []     
        self._last_alert_time = 0.0      # ADD THIS
        self._cooldown = 5.0   # [(track_id, timestamp)]

    def update(self, track_id: int, cx: float, cy: float) -> Optional[Alert]:
        x1, y1, x2, y2 = self.roi
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            return None

        now = time.time()
        self.entries.append((track_id, now))

        # Purge old entries
        self.entries = [(tid, t) for tid, t in self.entries if now - t < self.window]

        unique_ids = {tid for tid, _ in self.entries}
        if len(unique_ids) >= 2:
            now2 = time.time()
            if now2 - self._last_alert_time < self._cooldown:
                self.entries.clear()
                return None
            self._last_alert_time = now2
            alert = Alert(
                kind="ticket_abuse",
                 camera_id=0,
                track_ids=list(unique_ids),
                timestamp=now,
                confidence=min(1.0, len(unique_ids) * 0.4),
            )
            self.entries.clear()
            return alert
        return None

class StampedeDetector:
    """
    Fires when a crowd shows simultaneous lean + unidirectional movement.
    Uses pose torso angles + average tracker velocity (replacing optical flow).
    """

    def __init__(self, tilt_thresh=28.0, crowd_frac=0.55, flow_mag=3.5):
        self.tilt_thresh  = tilt_thresh   # degrees from vertical
        self.crowd_frac   = crowd_frac    # fraction of crowd that must be tilted
        self.flow_mag     = flow_mag      # min velocity magnitude
        self._flow_vec    = (0.0, 0.0)

    def update_flow(self, active_states: list[PersonState]):
        """Calculates crowd flow using ByteTrack velocities instead of dense optical flow."""
        vx, vy, count = 0.0, 0.0, 0
        now = time.time()
        for state in active_states:
            # Need at least 2 points to compute velocity, taken within last 1.5 seconds
            recent_pos = [(cx, cy, t) for cx, cy, t in state.positions if now - t < 1.5]
            if len(recent_pos) >= 5: # Require at least 5 frames of history
                p1 = recent_pos[0]
                p2 = recent_pos[-1]
                dt = p2[2] - p1[2]
                if dt > 0:
                    vx += (p2[0] - p1[0]) / dt
                    vy += (p2[1] - p1[1]) / dt
                    count += 1
        
        if count > 0:
            # Average pixels per second, normalized to match old optical flow scale (~30 fps)
            self._flow_vec = ((vx / count) / 30.0, (vy / count) / 30.0)
        else:
            self._flow_vec = (0.0, 0.0)

    def check(self, angles: list[Optional[float]], camera_id: int) -> Optional[Alert]:
        valid = [a for a in angles if a is not None]
        if len(valid) < 3:
            return None

        tilted = sum(1 for a in valid if a > self.tilt_thresh)
        frac   = tilted / len(valid)
        flow_mag = np.hypot(*self._flow_vec)

        if frac >= self.crowd_frac and flow_mag >= self.flow_mag:
            return Alert(
                kind="stampede",
                camera_id=camera_id,
                track_ids=[],
                timestamp=time.time(),
                confidence=round(min(1.0, frac * flow_mag / 10), 2),
            )
        return None


class VISTADetector:
    """Main pipeline: pose detection → tracking → alert engines."""

    def __init__(
        self,
        camera_id: int = 0,
        source: int | str = 0,
        gate_roi: tuple = (200, 300, 440, 480),
        conf: float = 0.6,
    ):
        self.camera_id  = camera_id
        self.source     = source
        self.conf       = conf
        self.output_path = None
        self.writer     = None

        self.model      = YOLO("yolov8n-pose.pt")
        self.model.to("cuda")
        print(f"[VISTA] YOLO running on: {next(self.model.model.parameters()).device}")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self.pose_ana   = PoseAnalyzer()
        self.ticket_det = TicketAbuseDetector(roi=gate_roi)
        self.stampede   = StampedeDetector()

        self.states: dict[int, PersonState] = defaultdict(lambda: PersonState(track_id=-1))
        self.alerts: list[Alert] = []
        self.reid = ReIDGallery(sim_threshold=0.75, ttl_sec=30.0)
        self.local_to_global: dict[int, int] = {}
        self.id_consensus: dict[int, list[int]] = defaultdict(list) # track_id -> [candidate_ids]

        # Annotators
        self.box_ann  = sv.BoxAnnotator(thickness=2)
        self.label_ann = sv.LabelAnnotator()
        self.pose_ann   = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=2)
        self.vertex_ann = sv.VertexAnnotator(color=sv.Color.RED, radius=4)
        self._frame_count = 0

    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Alert]]:
        self._frame_count += 1
        frame = cv2.resize(frame, (640, 360))
        frame_alerts = []

        results = self.model(frame, conf=self.conf, verbose=False, half=True)[0]

        # Convert to supervision detections
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        keypoints_all = results.keypoints.data.cpu().numpy()
        torso_angles = []

        tracker_ids = detections.tracker_id if detections.tracker_id is not None else []

        # Collect all crops for batch ReID
        crops = []
        valid_data = []
        for i, (box, track_id) in enumerate(zip(detections.xyxy, tracker_ids)):
            if track_id is None:
                continue
            
            x1b, y1b, x2b, y2b = map(int, box)
            w_box, h_box = x2b - x1b, y2b - y1b
            
            # Skip tiny detections early to maintain list synchronization
            if w_box < 32 or h_box < 64: 
                continue

            kps = keypoints_all[i] if i < len(keypoints_all) else None
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            
            # RAZOR SHARP: Full body crop with 5% padding
            # Quality Check: Only ReID if box is large enough for meaningful features
            if h_box < 100 or w_box < 40:
                continue

            pw, ph = int(w_box * 0.05), int(h_box * 0.05)
            crop = frame[max(0, y1b-ph):min(frame.shape[0], y2b+ph), 
                         max(0, x1b-pw):min(frame.shape[1], x2b+pw)]
            
            if crop.size > 0:
                crops.append(crop)
                valid_data.append((i, box, track_id, kps, cx, cy))

        # Batch ReID inference every 2 frames
        if crops and self._frame_count % 2 == 0:
            embeddings = self.reid._embed_batch(crops)
            # We will implement match_batch in reid.py for Hungarian assignment
            if hasattr(self.reid, "match_batch"):
                matches = self.reid.match_batch(valid_data, embeddings, self.camera_id)
                self.local_to_global.update(matches)
            else:
                # Fallback until reid.py is updated
                for idx, (_, _, track_id, _, _, _) in enumerate(valid_data):
                    gid = self.reid.update_with_embedding(int(track_id), embeddings[idx], self.camera_id)
                    self.local_to_global[track_id] = gid

        # Main processing loop
        for i, (box, track_id, kps, cx, cy) in enumerate([(d[1], d[2], d[3], d[4], d[5]) for d in valid_data]):
            # STICKY ID logic: If we already have a mapping for this track_id, keep it!
            # Only let match_batch override it if track_id is not in local_to_global.
            global_id = self.local_to_global.get(track_id, track_id)

            state = self.states[track_id]
            state.track_id = track_id
            state.positions.append((cx, cy, time.time()))
            if len(state.positions) > 30:
                state.positions.pop(0)

            alert = self.ticket_det.update(track_id, cx, cy)
            if alert:
                alert.frame = frame.copy()
                frame_alerts.append(alert)

            if kps is not None:
                angle = self.pose_ana.torso_angle(kps)
                torso_angles.append(angle)
                if self.pose_ana.is_fallen(kps, box):
                    frame_alerts.append(Alert(
                        kind="fall", camera_id=self.camera_id,
                        track_ids=[track_id], timestamp=time.time(),
                        confidence=0.85, frame=frame.copy()
                    ))

        active_states = [self.states[tid] for tid in tracker_ids if tid is not None]
        self.stampede.update_flow(active_states)
        s_alert = self.stampede.check(torso_angles, self.camera_id)
        if s_alert:
            frame_alerts.append(s_alert)

        # Annotate
        kp_data = sv.KeyPoints.from_ultralytics(results)
        annotated = self.pose_ann.annotate(frame.copy(), kp_data)
        annotated = self.vertex_ann.annotate(annotated, kp_data)
        annotated = self.box_ann.annotate(annotated, detections)

        labels = [f"G#{self.local_to_global.get(tid, tid)}"
                  for tid in (detections.tracker_id if detections.tracker_id is not None else [])]
        annotated = self.label_ann.annotate(annotated, detections, labels)

        # Gate ROI
        x1, y1, x2, y2 = self.ticket_det.roi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, "GATE ROI", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return annotated, frame_alerts

    def run(self, display: bool = True):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[VISTA] Camera {self.camera_id} started. Press Q to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self._frame_count % 2 != 0:
                self._frame_count += 1
                continue

            annotated, frame_alerts = self._process_frame(frame)

            for a in frame_alerts:
                print(f"[ALERT] {a.kind.upper()} | cam={a.camera_id} | ids={a.track_ids} | conf={a.confidence:.2f}")

            if display:
                annotated = cv2.resize(annotated, (640, 360))
                cv2.imshow("VISTA", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()
        
    def run(self, display: bool = True, output: Optional[str] = None):
        cap = cv2.VideoCapture(self.source)
        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Get frame size from cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # We resize to 960x540 in our display logic, let's save at that res for clarity
            self.writer = cv2.VideoWriter(output, fourcc, 20.0, (960, 540))
            print(f"[VISTA] Recording output to {output}...")

        print(f"[VISTA] Camera {self.camera_id} started. Press Q to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, frame_alerts = self._process_frame(frame)
            self.alerts.extend(frame_alerts)

            # Resize for display/output
            annotated_resized = cv2.resize(annotated, (960, 540))

            if self.writer:
                self.writer.write(annotated_resized)

            if display:
                cv2.imshow("VISTA", annotated_resized)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0, help="Video path or camera index")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--gate-roi", nargs=4, type=int, default=[200, 300, 440, 480],
                   metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--output", default=None, help="Path to save output video")
    args = p.parse_args()

    detector = VISTADetector(
        camera_id=args.camera_id,
        source=args.source if args.source != "0" else 0,
        gate_roi=tuple(args.gate_roi),
    )
    detector.run(output=args.output)
