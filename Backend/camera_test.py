import cv2
import time

def test_cameras():
    for i in range(5):
        print(f"Testing index {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Index {i} opened! Capture 5 frames...")
            for j in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"Frame {j} mean brightness: {frame.mean()}")
                else:
                    print(f"Frame {j} failed to read")
                time.sleep(0.1)
            cap.release()
        else:
            print(f"Index {i} failed to open")

if __name__ == "__main__":
    test_cameras()
