import os
import cv2
import numpy as np
import base64
import logging
import math
import threading
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# Set logging to error only
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import gc

# Global CORS
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# Resilient paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_26_PATH = os.path.join(BASE_DIR, 'landmark_cnn_model.h5')
MODEL_8_PATH = os.path.join(BASE_DIR, 'cnn8grps_rad1_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.pickle')

# Global Initialization (Prevents memory leaks/hangs)
model = None
detector = None
labels = None
is_26_class = False
lock = threading.Lock()

def init_ai():
    global model, detector, labels, is_26_class
    try:
        if os.path.exists(MODEL_26_PATH) and os.path.exists(LABELS_PATH):
            model = load_model(MODEL_26_PATH)
            with open(LABELS_PATH, 'rb') as f:
                labels = pickle.load(f)
            detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.4, modelComplexity=1)
            is_26_class = True
            print("🚀 AI SUCCESS: 26-Class Model Ready")
        elif os.path.exists(MODEL_8_PATH):
            model = load_model(MODEL_8_PATH)
            detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.4)
            is_26_class = False
            print("📦 AI FALLBACK: 8-Group Model Ready")
    except Exception as e:
        print(f"❌ AI ERROR: Initialization failed: {e}")

# Run once on load
init_ai()

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def get_refinement(ch1, pts):
    """
    HEURISTIC REFINEMENT: Correcting CNN mistakes using geometry.
    This ensures E/S/T, B/D/I, P/Q/Z etc. are always mathematically correct.
    """
    # 1. The Fist Group (A, E, M, N, S, T)
    fist_letters = ['A', 'E', 'M', 'N', 'S', 'T']
    if ch1 in fist_letters:
        # Check if actually a fist
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            # Refine within the fist group
            if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]: return 'A'
            if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]: return 'E'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]: return 'N'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]: return 'M'
            if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][1] < pts[14][1]: return 'T'
            return 'S'

    # 2. Pointing/Curve Group (C, O, G, H)
    if ch1 in ['C', 'O']:
        if distance(pts[12], pts[4]) > 42: return 'C'
        return 'O'
    if ch1 in ['G', 'H']:
        if distance(pts[8], pts[12]) > 65: return 'G'
        return 'H'

    # 3. Down-Signs (P, Q, Z)
    if ch1 in ['P', 'Q', 'Z']:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]: return 'Z'
            else: return 'Q'
        return 'P'

    # 4. Open-Finger Group (B, D, F, I, K, R, U, V, W)
    if ch1 in ['B', 'D', 'F', 'I', 'K', 'R', 'U', 'V', 'W']:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): return 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): return 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): return 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]): return 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]): return 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1]<pts[9][1]: return 'K'
        if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): return 'R'

    # 5. Little/Thumb (Y, J)
    if ch1 in ['Y', 'J']:
        if distance(pts[8], pts[4]) > 42: return 'Y'
        return 'J'
        
    return ch1 # Fallback to original prediction

@app.before_request
def log_request_info():
    print(f"DEBUG: Request to {request.path}")

@app.route('/predict', methods=['POST'])
def predict():
    print(f"DEBUG: Processing /predict request")
    try:
        with lock:
            if model is None: return jsonify({'prediction': 'No Model'})
            data = request.json
            img_data = base64.b64decode(data['image'].split(',')[1])
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            
            img = cv2.flip(img, 1)
            hands, _ = detector.findHands(img, draw=False, flipType=True)
            
            if not hands: return jsonify({'prediction': '...', 'status': 'No Hand'})
            
            hand = hands[0]
            x, y, w, h = hand['bbox']
            pts = hand['lmList']
            
            # Skeleton Processing
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            ox, oy = ((400 - w) // 2) - x, ((400 - h) // 2) - y
            def p(idx): return (pts[idx][0] + ox, pts[idx][1] + oy)

            for t in [range(0, 4), range(5, 8), range(9, 12), range(13, 16), range(17, 20)]:
                for i in t: cv2.line(white, p(i), p(i+1), (0, 255, 0), 3)
            for p1, p2 in [(5,9), (9,13), (13,17), (0,5), (0,17)]:
                cv2.line(white, p(p1), p(p2), (0, 255, 0), 3)

            # AI Logic
            if is_26_class:
                img_input = cv2.resize(white, (128, 128)).reshape(1, 128, 128, 3) / 255.0
                prob = model.predict(img_input, verbose=0)[0]
                char = str(labels[np.argmax(prob)])
                conf = np.max(prob)
            else:
                img_input = white.reshape(1, 400, 400, 3)
                prob = model.predict(img_input, verbose=0)[0]
                groups = ['A', 'B', 'C', 'G', 'L', 'P', 'X', 'Y'] # Group bases
                char = groups[np.argmax(prob)]
                conf = np.max(prob)

            # Geometric Refinement (THE FIX)
            # This corrects the AI's "sometimes wrong" guesses using math.
            char = get_refinement(char, pts) if conf > 0.3 else "..."

            _, buffer = cv2.imencode('.jpg', white, [cv2.IMWRITE_JPEG_QUALITY, 50])
            skeleton_b64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'prediction': char,
                'status': 'Success',
                'conf': float(conf),
                'skeleton': skeleton_b64
            })
    except Exception as e:
        return jsonify({'error': str(e), 'prediction': '...'})

@app.route('/get_sign/<char>')
def get_sign(char):
    char = char.lower()
    # 1. Try static_signs (Render-friendly)
    img_path = os.path.join(BASE_DIR, 'static_signs', f'{char}.jpg')
    
    # 2. Fallback to local full dataset
    if not os.path.exists(img_path):
        img_path = os.path.join(BASE_DIR, 'AtoZ_3.1', char.upper(), f'{char.upper()}.jpg')

    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({'status': 'success', 'image': b64_img})
    
    return jsonify({'status': 'error', 'message': 'Sign not found'}), 404

@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "message": "SignVoiceAI API is running",
        "model_loaded": MODEL_26_PATH is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)
