import cv2
import numpy as np
import time
from detect_and_track import ObjectDetectorTracker
from controller import AirSimController, TargetSelector

def smoke_test():
    print("Running Robust Tracking Smoke Test...")
    
    # 1. Initialize Tracker
    # Note: We use a small model or mock it if needed
    try:
        tracker = ObjectDetectorTracker(model_name='yolov8n.pt')
        print("✅ Tracker initialized")
    except Exception as e:
        print(f"❌ Tracker initialization failed: {e}")
        return

    # 2. Mock a frame (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Smoke Test Frame", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 3. Test Detection & Tracking
    print("Testing detect_and_track loop...")
    try:
        # First pass - might just detect
        annotated = tracker.detect_and_track(frame)
        print("✅ First pass completed")
        
        # Second pass - should have tracks (if detections exist)
        # We'll rely on the model detecting 'something' in the mock frame or just check it doesn't crash
        annotated = tracker.detect_and_track(frame)
        print("✅ Second pass completed")
        
    except Exception as e:
         print(f"❌ Error during tracking loop: {e}")
         import traceback
         traceback.print_exc()

    # 4. Test Controller compute_control
    print("Testing Controller logic...")
    controller = AirSimController()
    # Mock a primary target
    mock_target = {
        'id': 1,
        'bbox': [100, 100, 200, 250], # width=100, height=150
        'confidence': 0.9,
        'class_name': 'person'
    }
    vx, vy, vz, yaw = controller.compute_control(mock_target, 640, 480)
    print(f"✅ Control signals: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw={yaw:.2f}")
    
    # 5. Test Target Selection
    print("Testing Selection logic...")
    selector = TargetSelector()
    mock_tracks = [
        {'id': 1, 'bbox': [0,0,10,10], 'class_name': 'car'},
        {'id': 2, 'bbox': [100,100,150,150], 'class_name': 'pedestrian'}
    ]
    best = selector.select(mock_tracks)
    if best and best['class_name'] == 'pedestrian':
        print("✅ Target Selection prioritized 'pedestrian' correctly")
    else:
        print("❌ Target Selection failed priority check (Found: " + str(best.get('class_name') if best else 'None') + ")")

    print("\nSmoke Test Finished!")

if __name__ == "__main__":
    smoke_test()
