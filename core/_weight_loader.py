"""
OSNet weight loader — tries multiple sources in order.
Imported by reid.py — do not run directly.
"""

import os
import urllib.request
import torch

# Candidate weight files — tried in order
_CANDIDATES = [
    # 1. MSMT17 pretrained (best for cross-camera, diverse scenes)
    os.path.join(os.path.dirname(__file__), "osnet_x1_0_msmt17.pth"),
    # 2. Market-1501 pretrained (standard benchmark)
    os.path.join(os.path.dirname(__file__), "osnet_x1_0_market.pth"),
    # 3. Any .pth in core/ that looks like OSNet
    *[
        os.path.join(os.path.dirname(__file__), f)
        for f in os.listdir(os.path.dirname(__file__))
        if f.endswith(".pth") and "osnet" in f.lower()
    ],
]

# Direct download sources — tried if no local file found
_DOWNLOAD_SOURCES = [
    # HuggingFace — fast-reid model zoo mirror (no auth)
    {
        "url":  "https://huggingface.co/JDAI-CV/fast-reid/resolve/main/osnet_x1_0_imagenet.pth",
        "dest": os.path.join(os.path.dirname(__file__), "osnet_x1_0_market.pth"),
        "name": "OSNet-x1.0 (ImageNet init via HuggingFace)",
    },
]


def _reporthook(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%  ", end="", flush=True)


def find_or_download_weights(device: torch.device) -> str | None:
    """
    Returns path to a valid .pth file, or None if everything fails.
    Tries local files first, then auto-downloads.
    """
    # 1. Local candidates
    seen = set()
    for path in _CANDIDATES:
        if path in seen or not os.path.exists(path):
            seen.add(path)
            continue
        seen.add(path)
        size = os.path.getsize(path)
        if size > 1_000_000:   # must be >1 MB — not a partial download
            print(f"[ReID] Found weights: {path} ({size/1e6:.1f} MB)")
            return path
        else:
            print(f"[ReID] Skipping {path} — too small ({size} bytes), likely corrupt")

    # 2. Auto-download
    print("[ReID] No local weights found. Attempting auto-download...")
    for src in _DOWNLOAD_SOURCES:
        try:
            print(f"  Source: {src['name']}")
            os.makedirs(os.path.dirname(src["dest"]), exist_ok=True)
            urllib.request.urlretrieve(src["url"], src["dest"], _reporthook)
            print()
            size = os.path.getsize(src["dest"])
            if size > 1_000_000:
                print(f"  ✓ Downloaded → {src['dest']} ({size/1e6:.1f} MB)")
                return src["dest"]
            else:
                print(f"  ✗ Download too small ({size} bytes) — skipping")
                os.remove(src["dest"])
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # 3. gdown fallback (MSMT17 — best weights)
    try:
        import gdown
        dest = os.path.join(os.path.dirname(__file__), "osnet_x1_0_market.pth")
        gdrive_id = "112EMUfBPYeYg70w-syK6V6Mx8-iFmH5q"
        print(f"  Trying gdown (MSMT17 weights)...")
        gdown.download(id=gdrive_id, output=dest, quiet=False)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"  ✓ gdown success → {dest}")
            return dest
    except ImportError:
        print("  gdown not installed (pip install gdown to enable)")
    except Exception as e:
        print(f"  gdown failed: {e}")

    print("""
[ReID] ⚠ Could not load pretrained OSNet weights.
       Running with RANDOM INIT — ReID will not work correctly!

       To fix, run ONE of these:
         pip install gdown && python download_weights.py
         OR manually download from:
           https://drive.google.com/file/d/112EMUfBPYeYg70w-syK6V6Mx8-iFmH5q
         Save to: VISTA/core/osnet_x1_0_market.pth
""")
    return None