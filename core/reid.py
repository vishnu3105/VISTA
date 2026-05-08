import cv2
import numpy as np
import torch
import torchreid
from dataclasses import dataclass, field
from typing import Optional
import time

import os
_BASE_DIR = os.path.dirname(__file__)
WEIGHTS = os.path.join(_BASE_DIR, "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth")

@dataclass
class TrackEntry:
    global_id: int
    camera_id: int
    local_track_id: int
    embeddings: list
    last_seen: float = field(default_factory=time.time)
    last_pos: Optional[tuple[float, float]] = None
    crop: Optional[np.ndarray] = None


class ReIDGallery:
    def __init__(self, sim_threshold: float = 0.75, ttl_sec: float = 30.0):
        self.sim_threshold = sim_threshold
        self.ttl           = ttl_sec
        self.gallery: list[TrackEntry] = []
        self._next_gid     = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=751,
            pretrained=False
        )
        torchreid.utils.load_pretrained_weights(self.model, WEIGHTS)
        self.model.to(self.device).eval().half()
        print(f"[ReID] OSNet x1_0 (torchreid) ready on {self.device} (FP16)")

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        img_yuv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = self.clahe.apply(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        img = cv2.resize(img, (128, 256))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        img  = torch.from_numpy(img).permute(2, 0, 1).half()
        return img

    def _embed_batch(self, crops: list) -> np.ndarray:
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                tensors.append(torch.zeros(3, 256, 128))
            else:
                tensors.append(self._preprocess(crop))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self.model(batch)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.cpu().numpy()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _avg_embedding(self, embeddings: list) -> np.ndarray:
        avg = np.mean(embeddings, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-6)

    def _purge_old(self):
        now = time.time()
        self.gallery = [e for e in self.gallery if now - e.last_seen < self.ttl]

    def update_with_embedding(self, local_track_id: int, embedding: np.ndarray, camera_id: int) -> int:
        self._purge_old()
        best_sim, best_entry = -1.0, None
        for entry in self.gallery:
            sim = self._cosine_sim(embedding, self._avg_embedding(entry.embeddings))
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        if best_sim >= self.sim_threshold and best_entry is not None:
            best_entry.embeddings.append(embedding)
            if len(best_entry.embeddings) > 15:
                best_entry.embeddings.pop(0)
            best_entry.last_seen      = time.time()
            best_entry.local_track_id = local_track_id
            best_entry.camera_id      = camera_id
            return best_entry.global_id
        else:
            gid = self._next_gid
            self._next_gid += 1
            self.gallery.append(TrackEntry(
                global_id=gid,
                camera_id=camera_id,
                local_track_id=local_track_id,
                embeddings=[embedding],
            ))
            return gid

    def match_batch(self, valid_data: list, embeddings: np.ndarray, camera_id: int) -> dict[int, int]:
        """RAZOR SHARP: Hungarian assignment for frame-level uniqueness."""
        from scipy.optimize import linear_sum_assignment
        self._purge_old()
        num_det = len(embeddings)
        num_gal = len(self.gallery)
        if num_det == 0: return {}
        if num_gal == 0:
            return {d[2]: self.update_with_embedding(d[2], embeddings[i], camera_id) 
                    for i, d in enumerate(valid_data)}

        # 1. Similarity Matrix
        gal_embs = np.stack([self._avg_embedding(e.embeddings) for e in self.gallery])
        sims = np.dot(embeddings, gal_embs.T) # (num_det, num_gal)
        
        # 2. Cost Matrix with Spatial Prior
        # valid_data[i] = (idx, box, track_id, kps, cx, cy)
        cost_matrix = np.zeros((num_det, num_gal))
        for i in range(num_det):
            cx, cy = valid_data[i][4], valid_data[i][5]
            for j in range(num_gal):
                sim = sims[i, j]
                cost = 1.0 - sim
                entry = self.gallery[j]
                
                # Spatial Anchor: Reward continuity, heavily penalize 'teleportation'
                if entry.last_pos and entry.camera_id == camera_id:
                    dist = np.hypot(cx - entry.last_pos[0], cy - entry.last_pos[1])
                    if dist < 50:
                        cost -= 0.15 # Spatial Bonus: strongly encourages keeping ID during collisions
                    elif dist > 250:
                        cost += 0.4
                
                # RAZOR SHARP: Temporal Decay (IDs get 'colder' over time)
                time_gap = time.time() - entry.last_seen
                if time_gap > 2.0:
                    cost += min(0.3, 0.05 * time_gap) # Max 0.3 penalty for being gone
                
                cost_matrix[i, j] = cost

        # 3. Hungarian Matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 4. Final Assignment
        results = {}
        matched_det = set()
        for r, c in zip(row_ind, col_ind):
            effective_sim = 1.0 - cost_matrix[r, c]
            is_cross_camera = (self.gallery[c].camera_id != camera_id)
            threshold = self.sim_threshold - 0.05 if is_cross_camera else self.sim_threshold
            
            if effective_sim >= threshold:
                entry = self.gallery[c]
                # Anti-feature drift: Only append if raw visual similarity is strong
                if sims[r, c] >= 0.80:
                    entry.embeddings.append(embeddings[r])
                    if len(entry.embeddings) > 20: entry.embeddings.pop(0)
                entry.last_seen = time.time()
                entry.last_pos = (valid_data[r][4], valid_data[r][5])
                entry.local_track_id = valid_data[r][2]
                entry.camera_id = camera_id
                results[entry.local_track_id] = entry.global_id
                matched_det.add(r)

        # 5. New Tracks (RAZOR SHARP: Strictly create new entries to avoid same-frame collisions)
        for i in range(num_det):
            if i not in matched_det:
                tid = valid_data[i][2]
                # Check if this track was recently assigned to a global_id
                # but got displaced by a better match this frame.
                # If so, it MUST get a new ID to maintain uniqueness.
                gid = self._next_gid
                self._next_gid += 1
                self.gallery.append(TrackEntry(
                    global_id=gid,
                    camera_id=camera_id,
                    local_track_id=tid,
                    embeddings=[embeddings[i]],
                    last_seen=time.time(),
                    last_pos=(valid_data[i][4], valid_data[i][5])
                ))
                results[tid] = gid

        return results

    def get_all(self) -> list[TrackEntry]:
        self._purge_old()
        return list(self.gallery)

    def cross_camera_matches(self) -> list[tuple]:
        matches = []
        entries = self.get_all()
        for i, a in enumerate(entries):
            for b in entries[i+1:]:
                if a.camera_id == b.camera_id:
                    continue
                sim = self._cosine_sim(
                    self._avg_embedding(a.embeddings),
                    self._avg_embedding(b.embeddings)
                )
                if sim >= self.sim_threshold:
                    matches.append((a, b, sim))
        return matches