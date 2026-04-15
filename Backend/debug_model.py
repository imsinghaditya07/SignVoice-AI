"""
debug_model.py
Run this to verify the model preprocessing is working correctly.
Watch the terminal — predicted class index should CHANGE as you
show different hand signs. If it never changes, preprocessing is wrong.

Model: cnn8grps_rad1_model.h5
  - Input:  (None, 400, 400, 3)  — color BGR image
  - Output: (None, 8)            — 8 groups
  - Groups: [0->AEMNST][1->BFDIUVWKR][2->CO][3->GH][4->L][5->PQZ][6->X][7->YJ]
  - Training used raw pixel values (0-255), NOT normalized!
"""
import cv2
import numpy as np
import math
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

model = tf.keras.models.load_model("cnn8grps_rad1_model.h5")
H, W = model.input_shape[1], model.input_shape[2]
CHANNELS = model.input_shape[3]

GROUP_NAMES = {
    0: "AEMNST", 1: "BFDIUVWKR", 2: "CO", 3: "GH",
    4: "L", 5: "PQZ", 6: "X", 7: "YJ"
}

print(f"Model input : {model.input_shape}")
print(f"Model output: {model.output_shape}")
print(f"Groups: {GROUP_NAMES}")
print("Press ESC to quit. Watch class index change with hand signs.\n")

hd = HandDetector(detectionCon=0.4, maxHands=1)
hd2 = HandDetector(detectionCon=0.4, maxHands=1)

OFFSET = 29

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

cap = cv2.VideoCapture(0)
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    hands, _ = hd.findHands(frame, draw=False)
    label = "No hand detected"
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        img_h, img_w, _ = frame.shape
        y1, y2 = max(0, y - OFFSET), min(img_h, y + h + OFFSET)
        x1, x2 = max(0, x - OFFSET), min(img_w, x + w + OFFSET)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size > 0:
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            handz, _ = hd2.findHands(crop, draw=False)
            
            if handz:
                pts = handz[0]['lmList']
                ox, oy = ((400 - w) // 2) - 15, ((400 - h) // 2) - 15
                
                # Draw skeleton (same as training data)
                for t in range(0, 4):
                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(5, 8):
                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(9, 12):
                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(13, 16):
                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(17, 20):
                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[5][0]+ox, pts[5][1]+oy), (pts[9][0]+ox, pts[9][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0]+ox, pts[9][1]+oy), (pts[13][0]+ox, pts[13][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0]+ox, pts[13][1]+oy), (pts[17][0]+ox, pts[17][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+ox, pts[0][1]+oy), (pts[5][0]+ox, pts[5][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+ox, pts[0][1]+oy), (pts[17][0]+ox, pts[17][1]+oy), (0, 255, 0), 3)
                for i in range(21):
                    cv2.circle(white, (pts[i][0]+ox, pts[i][1]+oy), 2, (0, 0, 255), 1)
                
                # Prediction — NO /255.0 normalization (model trained on raw 0-255)
                img_input = white.reshape(1, H, W, CHANNELS)
                pred = model.predict(img_input, verbose=0)
                confidence = np.max(pred)
                predicted_index = np.argmax(pred)
                group_name = GROUP_NAMES.get(predicted_index, "?")
                
                frame_num += 1
                if frame_num % 15 == 0:
                    bar = " | ".join([f"G{i}({GROUP_NAMES[i]}):{v:.2f}" for i, v in enumerate(pred[0])])
                    print(f"[{frame_num:04d}] Group: {predicted_index} ({group_name}) "
                          f"| Conf: {confidence:.2f} | All: {bar}")
                
                label = f"Group: {predicted_index} ({group_name})  Conf: {confidence:.2f}"
                
                # Show the skeleton view
                cv2.imshow("Skeleton Input", white)
    
    # Overlay on frame
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Debug Feed", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
