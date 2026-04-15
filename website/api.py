"""
SignVoice AI - Real-time Inference Engine
----------------------------------------
This module handles real-time hand landmark processing, neural network classification (CNN),
and visual asset serving for the SignVoice AI web platform.

Author: Antigravity AI
Updated: 2026-04-15
"""

import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import math
import logging
import glob

# Standard professional logging configuration
logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app)

# Global model and detector initialization
MODEL_PATH = '../cnn8grps_rad1_model.h5'
model = load_model(MODEL_PATH)
hd1 = HandDetector(maxHands=1, detectionCon=0.4)
hd2 = HandDetector(maxHands=1, detectionCon=0.4)
OFFSET = 29

def distance(x, y):
    """Computes Euclidean distance between two 2D points."""
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

def subgroup_classify(ch1, ch2, pts):
    """
    Refines the first-level CNN classification using detailed MediaPipe landmark heuristics.
    This solves ambiguity between similar signs (e.g., A/E/M/N/S/T) by checking
    finger relative positions and distances.
    """
    l=[[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],[6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],[1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],[5,4],[7,0],[7,5],[7,2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 0

    l=[[2,2],[2,1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[5][0] < pts[4][0]): ch1 = 0

    l=[[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
    pl=[ch1,ch2]
    if pl in l:
        if (pts[0][0]>pts[8][0] and pts[0][0]>pts[4][0] and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and pts[5][0] > pts[4][0]: ch1 = 2

    l = [[6,0],[6,6],[6,2]]
    if pl in l:
        if distance(pts[8],pts[16]) < 52: ch1 = 2

    l = [[1,4],[1,5],[1,6],[1,3],[1,0]]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1]<pts[20][1] and pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]: ch1 = 3

    l=[[4,6],[4,1],[4,5],[4,3],[4,7]]
    if pl in l:
        if pts[4][0]>pts[0][0]: ch1 = 3

    l = [[5, 3],[5,0],[5,7], [5, 4], [5, 2],[5,1],[5,5]]
    if pl in l:
        if pts[2][1]+15<pts[16][1]: ch1 = 3

    l = [[6, 4], [6, 1], [6, 2]]
    if pl in l:
        if distance(pts[4],pts[11])>55: ch1 = 4

    l = [[1, 4], [1, 6],[1,1]]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]): ch1 = 4

    l = [[3, 6], [3, 4]]
    if pl in l:
        if (pts[4][0]<pts[0][0]): ch1 = 4

    l = [[2, 2], [2, 5],[2,4]]
    if pl in l:
        if (pts[1][0] < pts[12][0]): ch1 = 4

    l = [[3, 6],[3,5],[3,4]]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]) and pts[4][1]>pts[10][1]: ch1 = 5

    l = [[3,2],[3,1],[3,6]]
    if pl in l:
        if pts[4][1]+17>pts[8][1] and pts[4][1]+17>pts[12][1] and pts[4][1]+17>pts[16][1] and pts[4][1]+17>pts[20][1]: ch1 = 5

    l = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]
    if pl in l:
        if pts[4][0]>pts[0][0]: ch1 = 5

    l = [[0, 2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]
    if pl in l:
        if pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]: ch1 = 5

    l = [[5, 7],[5,2],[5,6]]
    if pl in l:
        if pts[3][0]<pts[0][0]: ch1 = 7

    l = [[4, 6],[4,2],[4,4],[4,1],[4,5],[4,7]]
    if pl in l:
        if pts[6][1] < pts[8][1]: ch1 = 7

    l = [[6, 7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]]
    if pl in l:
        if pts[18][1] > pts[20][1]: ch1 = 7

    l = [[0,4],[0,2],[0,3],[0,1],[0,6]]
    if pl in l:
        if pts[5][0]>pts[16][0]: ch1 = 6

    l = [[7, 2]]
    if pl in l:
        if pts[18][1] < pts[20][1]: ch1 = 6

    l = [[2, 1],[2,2],[2,6],[2,7],[2,0]]
    if pl in l:
        if distance(pts[8],pts[16])>50: ch1 = 6

    l = [[4, 6],[4,2],[4,1],[4,4]]
    if pl in l:
        if distance(pts[4], pts[11]) < 60: ch1 = 6

    l = [[1,4],[1,6],[1,0],[1,2]]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0: ch1 = 6

    l = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 1

    l = [[6, 1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7, 6],[4,6],[4,1],[4,2],[0, 2],[7, 1],[7, 4],[6, 6],[7, 2],[7, 5],[7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 1

    l = [[6, 1], [6, 0],[4,2],[4,1],[4,6],[4,4]]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 1

    l = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0]<pts[0][0]) and pts[4][1]>pts[14][1]): ch1 = 1

    l = [[4, 1], [4, 2],[4, 4]]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5],[3,6]]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1]<pts[4][1]): ch1 = 1

    l = [[6, 6],[6, 4], [6, 1],[6,2]]
    if pl in l:
        if pts[5][0]-pts[4][0]-15<0: ch1 = 1

    l = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7, 5],[7, 1],[7, 6],[7, 7]]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])): ch1 = 1

    l = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]
    if pl in l:
        if (pts[4][0]<pts[5][0]+15) and ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])): ch1 = 7

    l = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1]>pts[14][1]: ch1 = 1

    l = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]
    fg=13
    if pl in l:
        if not(pts[0][0]+fg < pts[8][0] and pts[0][0]+fg < pts[12][0] and pts[0][0]+fg < pts[16][0] and pts[0][0]+fg < pts[20][0]) and not(pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50: ch1 = 1

    l = [[5, 0], [5, 5],[0,1]]
    if pl in l:
        if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1]: ch1 = 1

    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]: ch1 = 'A'
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]: ch1 = 'T'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]: ch1 = 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]: ch1 = 'M'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]: ch1 = 'N'

    if ch1 == 2:
        if distance(pts[12], pts[4]) > 42: ch1 = 'C'
        else: ch1 = 'O'

    if ch1 == 3:
        if (distance(pts[8], pts[12])) > 72: ch1 = 'G'
        else: ch1 = 'H'

    if ch1 == 7:
        if distance(pts[8], pts[4]) > 42: ch1 = 'Y'
        else: ch1 = 'J'

    if ch1 == 4: ch1 = 'L'
    if ch1 == 6: ch1 = 'X'

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]: ch1 = 'Z'
            else: ch1 = 'Q'
        else: ch1 = 'P'

    if ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] >pts[20][1]): ch1 = 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]): ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1]<pts[9][1]: ch1 = 'K'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 'U'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] >pts[9][1]): ch1 = 'V'
        if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): ch1 = 'R'

    return ch1


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image'})
            
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        hands, _ = hd1.findHands(img, draw=False, flipType=False)
        
        if not hands:
            return jsonify({'prediction': '...', 'status': 'No hands detected'})
            
        hand = hands[0]
        x, y, w, h = hand['bbox']
        img_h, img_w, _ = img.shape
        
        y1, y2 = max(0, y - OFFSET), min(img_h, y + h + OFFSET)
        x1, x2 = max(0, x - OFFSET), min(img_w, x + w + OFFSET)
        crop = img[y1:y2, x1:x2]
        
        if crop.size > 0:
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            handz, _ = hd2.findHands(crop, draw=False, flipType=False)
            if handz:
                pts = handz[0]['lmList']
                ox, oy = ((400 - w) // 2) - 15, ((400 - h) // 2) - 15
                
                for t in range(0, 4, 1): cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(5, 8, 1): cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(9, 12, 1): cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(13, 16, 1): cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                for t in range(17, 20, 1): cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[5][0]+ox, pts[5][1]+oy), (pts[9][0]+ox, pts[9][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0]+ox, pts[9][1]+oy), (pts[13][0]+ox, pts[13][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0]+ox, pts[13][1]+oy), (pts[17][0]+ox, pts[17][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+ox, pts[0][1]+oy), (pts[5][0]+ox, pts[5][1]+oy), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+ox, pts[0][1]+oy), (pts[17][0]+ox, pts[17][1]+oy), (0, 255, 0), 3)
                for i in range(21): cv2.circle(white, (pts[i][0]+ox, pts[i][1]+oy), 2, (0, 0, 255), 1)

                img_input = white.reshape(1, 400, 400, 3)
                prob = np.array(model.predict(img_input, verbose=0)[0], dtype='float32')
                
                ch1 = np.argmax(prob)
                prob_copy = prob.copy()
                prob_copy[ch1] = 0
                ch2 = np.argmax(prob_copy)
                
                confidence = np.max(prob)
                char = subgroup_classify(ch1, ch2, pts)
                
                _, buffer = cv2.imencode('.jpg', white)
                skeleton_b64 = base64.b64encode(buffer).decode('utf-8')

                if confidence > 0.70 and isinstance(char, str) and char != "...":
                    return jsonify({'prediction': char, 'status': 'Success', 'conf': float(confidence), 'skeleton': skeleton_b64})
                else:
                    return jsonify({'prediction': '...', 'status': 'Low conf', 'skeleton': skeleton_b64})
        
        return jsonify({'prediction': '...', 'status': 'Crop failed'})
        
    except Exception as e:
        return jsonify({'error': str(e), 'prediction': '...'})

@app.route('/get_sign/<letter>', methods=['GET'])
def get_sign(letter):
    letter = letter.upper()
    if letter == ' ' or not letter.isalpha():
        return jsonify({'status': 'space'})
        
    folder = os.path.join('../AtoZ_3.1', letter)
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        files = glob.glob(os.path.join(folder, ext))
        if files:
            img_path = files[0]
            with open(img_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            return jsonify({'status': 'success', 'image': b64})
    
    return jsonify({'status': 'missing'})

if __name__ == '__main__':
    app.run(port=5002, debug=False)
