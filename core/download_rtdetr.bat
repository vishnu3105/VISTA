@echo off
cd /d C:\Users\vishnu\Desktop\VISTA\core

echo ============================================================
echo Downloading RT-DETR-X weights...
echo This may take a few minutes (weights are ~220 MB)
echo ============================================================
echo.

python -c "from ultralytics import RTDETR; model = RTDETR('rtdetr-x.pt'); print('Downloaded!')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! RT-DETR-X weights are ready!
    echo Run: python detector.py --source crowd.mp4 --display
    echo ============================================================
) else (
    echo.
    echo ERROR: Download failed!
)

pause
