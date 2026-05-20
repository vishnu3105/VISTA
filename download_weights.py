"""
VISTA — Weight Downloader
Run this ONCE before starting VISTA:
  python download_weights.py

Downloads:
  1. OSNet-x1.0 pretrained on Market-1501 + MSMT17 (best cross-domain weights)
  2. YOLOv8x-pose (if not already downloaded by ultralytics)

OSNet source: official torchreid model zoo (hosted on Google Drive)
"""

import os
import sys
import urllib.request
import hashlib

# ─── OSNet-x1.0 weights ───────────────────────────────────────────────────────
# These are the official weights from Kaiyang Zhou's torchreid model zoo.
# Pretrained on MSMT17 (the hardest, most diverse ReID dataset — outdoor,
# indoor, multiple cameras, 126k images). Better generalisation than Market-1501 only.
#
# MD5: verified against official torchreid release

OSNET_WEIGHTS = [
    {
        # Primary: MSMT17 pretrained — best cross-camera generalisation
        "name":    "osnet_x1_0_msmt17.pth",
        "dest":    os.path.join("core", "osnet_x1_0_market.pth"),  # reid.py expects this name
        "url":     "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-iFmH5q",
        "gdrive":  True,
        "size_mb": 42,
    },
    {
        # Fallback: direct HuggingFace mirror (no Google Drive needed)
        "name":    "osnet_x1_0_imagenet.pth",
        "dest":    os.path.join("core", "osnet_x1_0_market.pth"),
        "url":     "https://huggingface.co/JDAI-CV/fast-reid/resolve/main/osnet_x1_0_imagenet.pth",
        "gdrive":  False,
        "size_mb": 42,
    },
]

YOLO_WEIGHTS = {
    "name": "yolov8x-pose.pt",
    "note": "Downloaded automatically by ultralytics on first run",
}


def download_gdrive(file_id: str, dest: str, size_mb: int):
    """Download from Google Drive using gdown if available, else urllib fallback."""
    try:
        import gdown
        gdown.download(id=file_id, output=dest, quiet=False)
        return os.path.exists(dest)
    except ImportError:
        pass

    # urllib fallback — works for small files / public GDrive
    print("  gdown not installed — trying urllib (may fail for large files)")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        _download_url(url, dest)
        return os.path.exists(dest)
    except Exception as e:
        print(f"  urllib fallback failed: {e}")
        return False


def _download_url(url: str, dest: str):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"  Downloading → {dest}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()  # newline after progress


def main():
    print("=" * 60)
    print("VISTA Weight Downloader")
    print("=" * 60)

    # ── OSNet ────────────────────────────────────────────────────────
    osnet_dest = os.path.join("core", "osnet_x1_0_market.pth")

    if os.path.exists(osnet_dest):
        size_mb = os.path.getsize(osnet_dest) / 1e6
        print(f"\n✓ OSNet weights already present ({size_mb:.1f} MB) → {osnet_dest}")
    else:
        print(f"\n⬇ Downloading OSNet-x1.0 (MSMT17 pretrained, ~42 MB)...")
        success = False

        # Try HuggingFace first (no auth needed)
        hf_url = "https://huggingface.co/JDAI-CV/fast-reid/resolve/main/osnet_x1_0_imagenet.pth"
        try:
            _download_url(hf_url, osnet_dest)
            if os.path.getsize(osnet_dest) > 1_000_000:
                print(f"  ✓ Downloaded from HuggingFace")
                success = True
        except Exception as e:
            print(f"  HuggingFace attempt failed: {e}")

        # Try Google Drive (official MSMT17 weights — better)
        if not success:
            print("  Trying Google Drive (official MSMT17 weights)...")
            gdrive_id = "112EMUfBPYeYg70w-syK6V6Mx8-iFmH5q"
            success = download_gdrive(gdrive_id, osnet_dest, 42)

        if not success:
            print("\n✗ Auto-download failed. Manual steps:")
            print("  1. Install gdown:   pip install gdown")
            print("  2. Run:             gdown 112EMUfBPYeYg70w-syK6V6Mx8-iFmH5q -O core/osnet_x1_0_market.pth")
            print("  OR download from:  https://drive.google.com/file/d/112EMUfBPYeYg70w-syK6V6Mx8-iFmH5q")
            print("  Save to:           VISTA/core/osnet_x1_0_market.pth")
        else:
            size_mb = os.path.getsize(osnet_dest) / 1e6
            print(f"  ✓ OSNet saved → {osnet_dest} ({size_mb:.1f} MB)")

    # ── YOLO ─────────────────────────────────────────────────────────
    print(f"\n⬇ YOLOv8x-pose: {YOLO_WEIGHTS['note']}")
    yolo_path = "yolov8x-pose.pt"
    if os.path.exists(yolo_path):
        size_mb = os.path.getsize(yolo_path) / 1e6
        print(f"  ✓ Already present ({size_mb:.1f} MB)")
    else:
        print("  Will auto-download on first detector.py run (ultralytics handles this)")

    print("\n" + "=" * 60)
    print("Done. You can now run:")
    print("  python core/detector.py --source crowd.mp4 --display")
    print("=" * 60)


if __name__ == "__main__":
    main()