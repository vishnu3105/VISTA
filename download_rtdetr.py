import subprocess
import sys

print("=" * 70)
print("Downloading RT-DETR-X weights...")
print("=" * 70)
print()

# Download using ultralytics
cmd = [
    sys.executable, "-c",
    "from ultralytics import RTDETR; model = RTDETR('rtdetr-x.pt'); print('Downloaded!')"
]

result = subprocess.run(cmd, cwd=r"C:\Users\vishnu\Desktop\VISTA\core")

if result.returncode == 0:
    print()
    print("=" * 70)
    print("SUCCESS! RT-DETR-X weights downloaded to:")
    print("  C:\\Users\\vishnu\\Desktop\\VISTA\\core\\rtdetr-x.pt")
    print()
    print("Now run:")
    print("  python detector.py --source crowd.mp4 --display")
    print("=" * 70)
else:
    print("ERROR: Download failed")
    sys.exit(1)
