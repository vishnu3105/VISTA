import cv2
import numpy as np
import torch
import clip
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class TrackEntry:
    global_id: int
    camera_id: int
    local_track_id: int
    embeddings: list
    last_seen: float = field(default_factory=time.time)
    crop: Optional[np.ndarray] = None

class ReIDGallery:
    def __init__(self, sim_threshold: float = 0.88, ttl_sec: float = 60.0):
        self.sim_threshold = sim_threshold
        self.ttl           = ttl_sec
        self.gallery: list[TrackEntry] = []
        self._next_gid     = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        print(f"[ReID] CLIP ViT-B/32 ready on {self.device}")

    def _embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _avg_embedding(self, embeddings: list) -> np.ndarray:
        avg = np.mean(embeddings, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-6)

    def _purge_old(self):
        now = time.time()
        self.gallery = [e for e in self.gallery if now - e.last_seen < self.ttl]

    def update(self, camera_id: int, local_track_id: int, crop_bgr: np.ndarray) -> tuple[int, bool]:
        self._purge_old()
        embedding = self._embed(crop_bgr)

        best_sim, best_entry = -1.0, None
        for entry in self.gallery:
            sim = self._cosine_sim(embedding, self._avg_embedding(entry.embeddings))
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        print(f"[ReID] best_sim={best_sim:.3f} threshold={self.sim_threshold}")

        if best_sim >= self.sim_threshold and best_entry is not None:
            best_entry.embeddings.append(embedding)
            if len(best_entry.embeddings) > 8:
                best_entry.embeddings.pop(0)
            best_entry.last_seen      = time.time()
            best_entry.local_track_id = local_track_id
            best_entry.camera_id      = camera_id
            best_entry.crop           = crop_bgr
            print(f"[ReID] MATCH gid={best_entry.global_id}")
            return best_entry.global_id, False
        else:
            gid = self._next_gid
            self._next_gid += 1
            self.gallery.append(TrackEntry(
                global_id=gid,
                camera_id=camera_id,
                local_track_id=local_track_id,
                embeddings=[embedding],
                crop=crop_bgr,
            ))
            print(f"[ReID] NEW gid={gid}")
            return gid, True

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