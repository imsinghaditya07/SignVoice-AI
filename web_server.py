import os
import cv2
import numpy as np
import threading
import time
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import math

app = Flask(__name__)
CORS(app)

# --- CONFIG & INITIALIZATION ---
MODEL_PATH = 'cnn8grps_rad1_model.h5'
WHITE_IMG_PATH = 'white.jpg'
OFFSET = 29

# Create white background image if missing
if not os.path.exists(WHITE_IMG_PATH):
    white_bg = np.ones((400, 400, 3), np.uint8) * 255
    cv2.imwrite(WHITE_IMG_PATH, white_bg)

# Global State
current_prediction = "..."
sentence = ""
last_added_char = ""
last_char_time = 0

# Initialize Model & Hand Detector
print("Loading model...")
model = load_model(MODEL_PATH)
detector = HandDetector(maxHands=1)
engine = pyttsx3.init()

# Camera shared between threads
cap = cv2.VideoCapture(0)

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def get_prediction_char(ch1, ch2, pts):
    """Refined logic from the original scripts to pick the exact character"""
    # Simplified version of the complex original logic
    # [0->aemnst][1->bfdiuvwkr][2->co][3->gh][4->l][5->pqz][6->x][7->yj]
    
    if ch1 == 0: # aemnst
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0]: return 'A'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1]: return 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0]: return 'M'
        return 'S'
    
    if ch1 == 1: # bdfi...
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1]: return 'B'
        if pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1]: return 'D'
        return 'F'

    if ch1 == 2: return 'C' if distance(pts[12], pts[4]) > 42 else 'O'
    if ch1 == 3: return 'G' if distance(pts[8], pts[12]) > 72 else 'H'
    if ch1 == 4: return 'L'
    if ch1 == 5: return 'P' # Simplified
    if ch1 == 6: return 'X'
    if ch1 == 7: return 'Y' if distance(pts[8], pts[4]) > 42 else 'J'
    
    return "?"

def process_frames():
    global current_prediction, sentence, last_added_char, last_char_time
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue
            
        frame = cv2.flip(frame, 1)
        hands, frame = detector.findHands(frame, draw=True, flipType=True)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Ensure ROI is within bounds
            y1, y2 = max(0, y - OFFSET), min(frame.shape[0], y + h + OFFSET)
            x1, x2 = max(0, x - OFFSET), min(frame.shape[1], x + w + OFFSET)
            
            try:
                # Landmark handling like the original script
                pts = hand['lmList']
                white = cv2.imread(WHITE_IMG_PATH)
                
                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15
                
                # Draw landmarks on white image (Original feature extraction)
                # Skeleton drawing logic
                for t in range(0, 4): cv2.line(white, (pts[t][0]-x+os_x, pts[t][1]-y+os_y), (pts[t+1][0]-x+os_x, pts[t+1][1]-y+os_y), (0, 255, 0), 3)
                for t in range(5, 8): cv2.line(white, (pts[t][0]-x+os_x, pts[t][1]-y+os_y), (pts[t+1][0]-x+os_x, pts[t+1][1]-y+os_y), (0, 255, 0), 3)
                # (Remaining lines to follow similar logic if needed, but for MVP we use the hand image or a subset)
                
                # Reshape for model
                img_input = cv2.resize(white, (400, 400))
                img_input = img_input.reshape(1, 400, 400, 3)
                
                # Predict
                prob = model.predict(img_input, verbose=0)[0]
                ch1 = np.argmax(prob)
                prob[ch1] = 0
                ch2 = np.argmax(prob)
                
                char = get_prediction_char(ch1, ch2, pts)
                current_prediction = char
                
                # Add to sentence with cooldown
                current_time = time.time()
                if char != last_added_char and current_time - last_char_time > 2.0:
                    sentence += char
                    last_added_char = char
                    last_char_time = current_time
                    
            except Exception as e:
                print(f"Inference Error: {e}")
                
        else:
            current_prediction = "..."

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            # Add prediction overlay to video stream
            cv2.putText(frame, f"Prediction: {current_prediction}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "prediction": current_prediction,
        "sentence": sentence
    })

@app.route('/clear', methods=['POST'])
def clear():
    global sentence
    sentence = ""
    return "OK"

@app.route('/speak', methods=['POST'])
def speak():
    global sentence
    if sentence:
        engine.say(sentence)
        engine.runAndWait()
    return "OK"

if __name__ == '__main__':
    print("Starting Web Server at http://localhost:5000")
    # Run inference in a separate thread to keep stream smooth
    threading.Thread(target=process_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
