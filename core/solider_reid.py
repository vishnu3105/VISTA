"""
VISTA — SOLIDER ReID Extractor
Uses official SOLIDER-REID repo code — zero architecture mismatch.
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

# Add SOLIDER-REID to path — relative to core/
_SOLIDER_PATH = os.path.join(os.path.dirname(__file__), '..', 'SOLIDER-REID')
sys.path.insert(0, os.path.abspath(_SOLIDER_PATH))

from config import cfg
from model.make_model import make_model

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _preprocess(crop_bgr: np.ndarray) -> torch.Tensor:
    yuv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = _CLAHE.apply(yuv[:,:,0])
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    rgb = cv2.resize(rgb, (128, 256))
    return _TF(rgb)

def _quality(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0: return False
    h, w = crop.shape[:2]
    if w < 40 or h < 100: return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > 50 and 15 < gray.mean() < 240


class SOLIDERExtractor:
    def __init__(self, weight_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        cfg_file = os.path.join(_SOLIDER_PATH, 'configs', 'msmt17', 'swin_base.yml')
        cfg.merge_from_file(cfg_file)
        cfg.freeze()

        # Build model using their exact code
        self.model = make_model(
            cfg,
            num_class=4101,
            camera_num=15,
            view_num=1,
            semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT
        )
        self.model.load_param(weight_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[SOLIDER] Official model loaded from {weight_path}")
        print(f"[SOLIDER] Ready on {self.device}")

    @torch.no_grad()
    def embed_batch(self, crops: list) -> tuple:
        tensors, valid = [], []
        for crop in crops:
            ok = _quality(crop)
            tensors.append(_preprocess(crop) if ok else torch.zeros(3, 256, 128))
            valid.append(ok)
        batch = torch.stack(tensors).to(self.device)
        # SOLIDER inference — returns feat after BN neck
        out = self.model(batch, cam_label=0, view_label=0)
        embs = out[0] if isinstance(out, tuple) else out
        embs = F.normalize(embs, p=2, dim=1).float().cpu().numpy().astype(np.float32)
        for i, ok in enumerate(valid):
            if not ok: embs[i] = 0.0
        return embs, valid
