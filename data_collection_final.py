import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback
import time

# Configuration
DATA_DIR = "AtoZ_3.1"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    for i in range(65, 91):  # A-Z
        os.makedirs(os.path.join(DATA_DIR, chr(i)))

capture = cv2.VideoCapture(0)
# Corrected backends and properties
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hd = HandDetector(detectionCon=0.5, maxHands=1)
hd2 = HandDetector(detectionCon=0.5, maxHands=1)

c_dir = 'A'
offset = 29
step = 1
flag = False # Saving flag
suv = 0 # Snapshot counter

print(f"Data Collection Started. Current folder: {c_dir}")
print("Keys: [A] Start/Stop Saving, [N] Next Letter, [ESC] Quit")

while True:
    try:
        ret, frame = capture.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        # Standard white bg in memory
        white = np.ones((400, 400, 3), np.uint8) * 255
        
        hands, _ = hd.findHands(frame, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Safe cropping
            y1, y2 = max(0, y - offset), min(frame.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(frame.shape[1], x + w + offset)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                handz, _ = hd2.findHands(crop, draw=False)
                if handz:
                    pts = handz[0]['lmList']
                    ox, oy = ((400 - w) // 2) - 15, ((400 - h) // 2) - 15
                    
                    # Draw Bone Skeleton
                    connections = [range(0, 4), range(5, 8), range(9, 12), range(13, 16), range(17, 20)]
                    for conn in connections:
                        for t in conn:
                            cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                    
                    for p1, p2 in [(5,9), (9,13), (13,17), (0,5), (0,17)]:
                        cv2.line(white, (pts[p1][0]+ox, pts[p1][1]+oy), (pts[p2][0]+ox, pts[p2][1]+oy), (0, 255, 0), 3)

                    # Save this processed skeleton (The AI trains on this)
                    processed_skeleton = np.array(white)
                    
                    # Draw red dots for visual preview only
                    preview = np.array(white)
                    for i in range(21):
                        cv2.circle(preview, (pts[i][0]+ox, pts[i][1]+oy), 2, (0, 0, 255), 1)
                    
                    cv2.imshow("Skeleton-Preview", preview)

                    # Saving logic
                    if flag:
                        # Only save every 3rd frame to get variety but keep it fast
                        if step % 3 == 0:
                            folder_path = os.path.join(DATA_DIR, c_dir)
                            if not os.path.exists(folder_path): os.makedirs(folder_path)
                            
                            # Use timestamp to avoid overwriting
                            img_name = f"{int(time.time()*1000)}.jpg"
                            cv2.imwrite(os.path.join(folder_path, img_name), processed_skeleton)
                            suv += 1
                            print(f"Saved {suv}/180 for {c_dir}")
                            
                            if suv >= 180:
                                flag = False
                                print(f"Done with {c_dir}! Press 'N' for next.")
                        
                        step += 1

        # UI Overlay
        status = "SAVING" if flag else "READY"
        cv2.putText(frame, f"Letter: {c_dir} | Count: {suv} | Status: {status}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("Data-Collection", frame)
        
        key = cv2.waitKey(1)
        if key == 27: break # ESC
        
        if key == ord('n'): # Next letter
            suv = 0
            flag = False
            curr = ord(c_dir)
            if curr < 90: # Z
                c_dir = chr(curr + 1)
            else:
                c_dir = 'A'
            print(f"Switched to {c_dir}")

        if key == ord('a'): # Toggle saving
            flag = not flag
            if flag: print(f"Recording {c_dir}...")
            else: print("Paused Recording.")

    except Exception:
        print(traceback.format_exc())

capture.release()
cv2.destroyAllWindows()