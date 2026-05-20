# VISTA — Visual Intelligence Surveillance & Threat Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8x-TensorRT%20FP16-green)
![Tracking](https://img.shields.io/badge/Tracking-OC--SORT-orange)
![ReID](https://img.shields.io/badge/ReID-SOLIDER%20Swin--Base-purple)
![Backend](https://img.shields.io/badge/Backend-FastAPI%20%2B%20WebSocket-black?logo=fastapi)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)

> Real-time crowd surveillance system — detects, tracks, and re-identifies individuals across video streams.

---

## The Problem

Traditional CCTV systems are passive. They record footage but cannot:
- Automatically detect threats or abnormal behaviour
- Track individuals across multiple camera feeds
- Re-identify a person after they leave and re-enter frame
- Alert operators in real time

VISTA solves this with a production-grade ML pipeline running entirely on-device.

---

## Production Stack

| Component | Implementation | Status |
|-----------|---------------|--------|
| Detection | YOLOv8x-pose + TensorRT FP16 | Production |
| Tracking | OC-SORT | Production |
| Re-Identification | SOLIDER Swin-Base (MSMT17) | Production |
| Backend | FastAPI + WebSocket | Integrated |
| Frontend | React Dashboard | Integrated |

---

## Performance

| Metric | Before | After |
|--------|--------|-------|
| YOLO inference latency | 28ms | 11ms |
| Optimisation method | Baseline PyTorch | TensorRT FP16 export |
| Hardware | RTX 4060 | RTX 4060 |

TensorRT FP16 export cut inference time by **60%** on the same hardware.

---

## Architecture

```
Video Stream
     ↓
YOLOv8x-pose (TensorRT FP16)
     ↓ bounding boxes + keypoints
OC-SORT Tracker
     ↓ stable track IDs
SOLIDER Swin-Base ReID
     ↓ identity embeddings (FAISS indexed)
FastAPI + WebSocket
     ↓ real-time alerts
React Dashboard
```

---

## Key Engineering Decisions

**Why TensorRT over standard PyTorch inference?**
PyTorch inference on YOLOv8x-pose ran at 28ms per frame — too slow for real-time surveillance. TensorRT FP16 export reduced this to 11ms while maintaining detection accuracy.

**Why OC-SORT over ByteTrack or StrongSORT?**
- ByteTrack: IoU-only association — fails when people occlude each other
- StrongSORT: ECC overhead caused memory growth over long video streams
- OC-SORT: stable tracking latency, handles occlusion better, no memory leak

**Why SOLIDER Swin-Base over OSNet?**
SOLIDER trained on MSMT17 (large-scale multi-camera dataset) gave significantly better cross-camera re-identification than OSNet on real footage. Swin-Base architecture provides stronger feature extraction for ReID tasks.

---

## Project Structure

```
VISTA/
├── core/
│   ├── detector.py        # YOLOv8x TensorRT inference
│   ├── tracker.py         # OC-SORT implementation
│   ├── reid.py            # SOLIDER ReID + FAISS indexing
│   └── pipeline.py        # Unified detection-tracking-reid pipeline
├── api/
│   ├── main.py            # FastAPI + WebSocket server
│   └── schemas.py         # Pydantic models
├── frontend/              # React dashboard
├── .gitignore
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/vishnu3105/VISTA
cd VISTA
pip install -r requirements.txt

# Export YOLOv8x to TensorRT (requires NVIDIA GPU)
python -c "
from ultralytics import YOLO
m = YOLO('yolov8x-pose.pt')
m.export(format='engine', half=True, imgsz=480, device=0)
"

# Start backend
uvicorn api.main:app --reload

# Start frontend
cd frontend && npm install && npm start
```

**Requirements:** NVIDIA GPU with CUDA, TensorRT installed

---

## What's Next

- [ ] Cross-camera ReID — track individuals across multiple feeds
- [ ] RTSP stream integration — connect to real IP cameras
- [ ] Anomaly detection tuning — stampede, pushing, loitering alerts
- [ ] Deployment hardening — Docker + production config

---

## Team

Built by a 3-person team at Sri Sairam Institute of Technology, Chennai.
Core ML pipeline, tracking architecture, and ReID system engineered by **Vishnu N**.

[LinkedIn](https://www.linkedin.com/in/vishnu-n-7bb753320/) · [GitHub](https://github.com/vishnu3105)
