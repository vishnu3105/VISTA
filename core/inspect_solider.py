"""
Run from VISTA root:
python inspect_solider.py
"""
import torch

w = torch.load('core/swin_base_msmt17.pth', map_location='cpu', weights_only=False)
print("Type:", type(w))

if isinstance(w, dict):
    print("Top keys:", list(w.keys())[:10])
    for k, v in list(w.items())[:5]:
        if isinstance(v, dict):
            print(f"  '{k}' -> dict, {len(v)} keys, first 3:", list(v.keys())[:3])
        elif hasattr(v, 'shape'):
            print(f"  '{k}' -> tensor {v.shape}")
        else:
            print(f"  '{k}' -> {type(v).__name__} = {str(v)[:60]}")
