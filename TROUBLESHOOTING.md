# Troubleshooting Guide for YOLOv8 Live Object Detection & Tracking

## Common Issues and Solutions

### 1. OpenMP Runtime Error (libiomp5md.dll)

**Error Message:**

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**Solution:**
This error occurs when multiple libraries (PyTorch, OpenCV, NumPy) include their own OpenMP runtimes.

**Fixed automatically in the application**, but if you encounter this:

**Option A: Use the provided run scripts**

- Windows: `run.bat` or `run.ps1`
- Linux/macOS: `run.sh`

**Option B: Set environment variable manually**

```bash
# Windows Command Prompt
set KMP_DUPLICATE_LIB_OK=TRUE
python app.py

# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python app.py

# Linux/macOS
export KMP_DUPLICATE_LIB_OK=TRUE
python app.py
```

### 2. Camera Access Issues

**Error Messages:**

- "Could not open camera"
- "Cannot access camera"
- "Failed to read frame from webcam"

**Solutions:**

**A. Check camera permissions:**

- Ensure camera is not being used by another application
- Grant camera permissions to your terminal/Python

**B. Try different camera indices:**
Edit `config.py`:

```python
CAMERA_INDEX = 1  # Try 0, 1, 2, etc.
```

**C. Check available cameras:**

```python
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
```

### 3. Package Import Errors

**Error Messages:**

- "ModuleNotFoundError: No module named 'cv2'"
- "cannot import name 'DeepSort'"

**Solutions:**

**A. Reinstall dependencies:**

```bash
pip install -r requirements.txt
```

**B. For Python 3.13+ compatibility:**

```bash
pip install -r requirements-python313.txt
```

**C. Install packages individually:**

```bash
pip install flask ultralytics opencv-python deep-sort-realtime numpy torch torchvision scipy pillow
```

### 4. Performance Issues

**Symptoms:**

- Low FPS
- High CPU usage
- Laggy video stream

**Solutions:**

**A. Use a smaller YOLOv8 model:**
Edit `config.py`:

```python
YOLO_MODEL = 'yolov8n.pt'  # Fastest (nano)
# YOLO_MODEL = 'yolov8s.pt'  # Small
# YOLO_MODEL = 'yolov8m.pt'  # Medium
```

**B. Reduce camera resolution:**
Edit `config.py`:

```python
CAMERA_WIDTH = 480    # Reduce from 640
CAMERA_HEIGHT = 360   # Reduce from 480
```

**C. Increase confidence threshold:**
Edit `config.py`:

```python
CONFIDENCE_THRESHOLD = 0.7  # Higher = fewer detections
```

### 5. Web Browser Issues

**Symptoms:**

- Blank page
- "This site can't be reached"
- Video stream not loading

**Solutions:**

**A. Check server status:**

- Look for "Running on http://127.0.0.1:5000" in terminal
- Try both `localhost:5000` and `127.0.0.1:5000`

**B. Firewall/Antivirus:**

- Temporarily disable firewall
- Add Python to firewall exceptions

**C. Browser compatibility:**

- Try different browsers (Chrome, Firefox, Edge)
- Enable camera permissions in browser

### 6. Model Download Issues

**Error Messages:**

- "Failed to download model"
- Connection timeout

**Solutions:**

**A. Manual download:**
Download YOLOv8 model manually and place in project folder:

- [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)

**B. Check internet connection:**
Ensure stable internet for initial model download (~6MB)

### 7. Python Version Compatibility

**Error Messages:**

- "Python version not supported"
- Package installation failures

**Solutions:**

**A. Check Python version:**

```bash
python --version
```

**B. Upgrade Python:**

- Minimum required: Python 3.8
- Recommended: Python 3.9-3.11
- Latest tested: Python 3.13

### 8. Memory Issues

**Symptoms:**

- Application crashes
- "Out of memory" errors

**Solutions:**

**A. Close other applications:**

- Free up RAM
- Close unnecessary programs

**B. Reduce processing load:**

```python
# In config.py
TRAJECTORY_LENGTH = 10  # Reduce from 30
STREAM_FPS = 15         # Reduce from 30
```

## Getting Help

### 1. Run Validation Script

```bash
python validate_setup.py
```

### 2. Check System Requirements

- Python 3.8+
- 4GB+ RAM
- Webcam/Camera
- Internet (for initial setup)

### 3. Enable Debug Mode

Edit `config.py`:

```python
DEBUG = True
```

### 4. Check Logs

Look at terminal output for error messages and warnings.

## Quick Fixes

### Reset to Default Settings

1. Delete or rename `config.py`
2. Application will use default settings

### Clean Reinstall

1. Delete `.conda` folder (if exists)
2. Run: `pip uninstall -r requirements.txt -y`
3. Run: `pip install -r requirements.txt`

### Alternative Launch Methods

1. `python launch.py` - Auto-setup with validation
2. `python app.py` - Direct launch
3. `run.bat` - Windows batch file
4. `run.ps1` - PowerShell script

---

**If issues persist, check the terminal output for specific error messages and refer to the respective section above.**
