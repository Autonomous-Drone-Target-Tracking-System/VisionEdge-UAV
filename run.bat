@echo off
echo ===============================================
echo  YOLOv8 Live Object Detection and Tracking
echo ===============================================
echo.
echo Setting up environment...
set KMP_DUPLICATE_LIB_OK=TRUE
echo.
echo Starting the application...
echo.
echo Make sure you have:
echo 1. Installed Python 3.8+
echo 2. Installed dependencies: pip install -r requirements.txt
echo 3. Connected a webcam to your computer
echo.
echo The application will start on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ===============================================
echo.

python app.py

pause
