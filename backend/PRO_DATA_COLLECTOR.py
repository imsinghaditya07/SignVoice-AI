import cv2
import numpy as np
import os
import time
from cvzone.HandTrackingModule import HandDetector

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "AtoZ_3.1")
IMG_SIZE = 400
SAMPLES_PER_LETTER = 150
OFFSET = 29

# Ensure folders exist
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
for i in range(65, 91):
    char_dir = os.path.join(DATA_DIR, chr(i))
    if not os.path.exists(char_dir): os.makedirs(char_dir)

# Initialize Camera and Detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
detector = HandDetector(maxHands=1, detectionCon=0.5)

current_char_idx = 0
letters = [chr(i) for i in range(65, 91)]
collecting = False
count = 0

print("=== PRO SIGN DATA COLLECTOR ===")
print("Keys: [SPACE] Start/Pause, [N] Next Letter, [R] Reset Count, [ESC] Exit")

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    hands, _ = detector.findHands(frame, draw=False)
    
    # White background for skeleton
    white = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    char = letters[current_char_idx]
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Exact matching logic for centering and drawing
        pts = hand['lmList']
        ox, oy = ((IMG_SIZE - w) // 2) - x, ((IMG_SIZE - h) // 2) - y
        
        def p(idx): return (pts[idx][0] + ox, pts[idx][1] + oy)

        # Draw Golden Standard Skeleton
        for t in [range(0, 4), range(5, 8), range(9, 12), range(13, 16), range(17, 20)]:
            for i in t: cv2.line(white, p(i), p(i+1), (0, 255, 0), 3)
        for p1, p2 in [(5,9), (9,13), (13,17), (0,5), (0,17)]:
            cv2.line(white, p(p1), p(p2), (0, 255, 0), 3)

        # SAVING LOGIC
        if collecting and count < SAMPLES_PER_LETTER:
            file_path = f"{DATA_DIR}/{char}/{time.time()}.jpg"
            cv2.imwrite(file_path, white)
            count += 1
            if count >= SAMPLES_PER_LETTER:
                collecting = False
                print(f"Done with {char}!")

    # UI Feedback
    cv2.putText(frame, f"Letter: {char} ({count}/{SAMPLES_PER_LETTER})", (20, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "STATUS: RECORDING" if collecting else "STATUS: IDLE", (20, 90), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0) if collecting else (0, 0, 255), 2)
    
    cv2.imshow("Sign-Collector", frame)
    cv2.imshow("Skeleton-Input", white)
    
    key = cv2.waitKey(1)
    if key == 27: break
    if key == ord(' '): collecting = not collecting
    if key == ord('n'):
        current_char_idx = (current_char_idx + 1) % 26
        count = 0
        collecting = False
    if key == ord('r'):
        count = 0

cap.release()
cv2.destroyAllWindows()
