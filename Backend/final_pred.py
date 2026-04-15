# Importing Libraries
import numpy as np
import math
import cv2
import os, sys
import threading
import time
import pickle
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from PIL import Image, ImageTk

# Optional Spell Check
try:
    import enchant
    ddd = enchant.Dict("en-US")
    HAS_ENCHANT = True
except Exception:
    HAS_ENCHANT = False

# Configuration
OFFSET = 29
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

class Application:
    def __init__(self):
        self.vs = None
        self.frame = None
        self.camera_ready = False
        self.processed_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Dual-mode state
        self.model = None
        self.numbers_model = None
        self.current_mode = "ALPHABETS" # Can toggle to "NUMBERS"
        threading.Thread(target=self.load_model_bg, daemon=True).start()
        
        self.current_symbol = "..."
        self.sentence = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.running = True
        self.frame_count = 0
        
        self.hd = HandDetector(detectionCon=0.4, maxHands=1)
        self.hd2 = HandDetector(detectionCon=0.4, maxHands=1)
        
        self.prev_char = ""
        self.last_spoken = ""
        self.last_spoken_time = 0

        # UI
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion Pro")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1300x750")
        
        # Keyboard Command to Toggle
        self.root.bind("<m>", lambda e: self.toggle_mode())
        self.root.bind("<M>", lambda e: self.toggle_mode())
        
        self.setup_ui()

        # Command-line Mode Starting
        if len(sys.argv) > 1 and sys.argv[1].lower() == "numbers":
            self.toggle_mode()
            
        # Threads
        threading.Thread(target=self.camera_thread, daemon=True).start()
        threading.Thread(target=self.inference_thread, daemon=True).start()

        self.video_loop()

    def load_model_bg(self):
        """Load the correct 8-group CNN model and the new Numbers model"""
        model_path = 'cnn8grps_rad1_model.h5'
        print(f"Loading 8-group model from {model_path}...")
        
        if not os.path.exists(model_path):
            print(f"ERROR: {model_path} not found!")
        else:
            self.model = load_model(model_path)
            print(f"Alphabet Model loaded! Input: {self.model.input_shape}")
            
        num_model_path = 'numbers_model.h5'
        if os.path.exists(num_model_path):
            print(f"Loading Numbers model from {num_model_path}...")
            self.numbers_model = load_model(num_model_path)
            print("Numbers mode is ready!")
        else:
            print("Numbers model not found. (Numbers mode disabled until trained)")

    def setup_ui(self):
        tk.Label(self.root, text="Sign Language To Text Conversion (Advanced)", font=("Courier", 30, "bold")).place(x=60, y=5)
        self.panel = tk.Label(self.root) 
        self.panel.place(x=50, y=80, width=580, height=450)
        self.panel2 = tk.Label(self.root) 
        self.panel2.place(x=700, y=115, width=400, height=400)
        
        tk.Label(self.root, text="Detected Sign:", font=("Courier", 20, "bold")).place(x=700, y=80)
        self.lbl_char = tk.Label(self.root, text="...", font=("Courier", 40, "bold"), fg="#3b82f6")
        self.lbl_char.place(x=950, y=70)
        
        self.btn_mode = tk.Button(self.root, text="Mode: ALPHABETS", font=("Courier", 16, "bold"), bg="#10b981", fg="white", command=self.toggle_mode)
        self.btn_mode.place(x=700, y=20)

        tk.Label(self.root, text="Sentence :", font=("Courier", 20, "bold")).place(x=50, y=550)
        self.lbl_sentence = tk.Label(self.root, text=" ", font=("Courier", 25), wraplength=1000, anchor="w", bg="white", relief="sunken")
        self.lbl_sentence.place(x=250, y=550, width=1000, height=60)

        self.btn_sugs = []
        for i in range(4):
            btn = tk.Button(self.root, text=" ", font=("Courier", 18), width=12, bg="#f3f4f6")
            btn.place(x=300 + (i * 200), y=650)
            self.btn_sugs.append(btn)
        
        self.btn_sugs[0].config(command=lambda: self.apply_sug(0))
        self.btn_sugs[1].config(command=lambda: self.apply_sug(1))
        self.btn_sugs[2].config(command=lambda: self.apply_sug(2))
        self.btn_sugs[3].config(command=lambda: self.apply_sug(3))

        tk.Button(self.root, text="Clear", font=("Courier", 20), bg="#ef4444", fg="white", command=self.clear_fun).place(x=50, y=650)

    def toggle_mode(self):
        if self.current_mode == "ALPHABETS":
            self.current_mode = "NUMBERS"
            self.btn_mode.config(text="Mode: NUMBERS", bg="#f59e0b")
        else:
            self.current_mode = "ALPHABETS"
            self.btn_mode.config(text="Mode: ALPHABETS", bg="#10b981")
        self.clear_fun()

    def camera_thread(self):
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.vs = cap
                self.camera_ready = True
                while self.running:
                    success, img = self.vs.read()
                    if success: self.frame = img
                    time.sleep(0.01)
                return

    def subgroup_classify(self, ch1, ch2, pts):
        """
        Apply the original subgroup heuristic logic from prediction_wo_gui.py.
        The 8-group CNN predicts one of 8 groups:
        [0->AEMNST][1->BFDIUVWKR][2->CO][3->GH][4->L][5->PQZ][6->X][7->YJ]
        Then landmark-based rules narrow it down to specific letters.
        """
        # ---- Group refinement (inter-group corrections) ----
        
        # condition for [Aemnst]
        l=[[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],[6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],[1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],[5,4],[7,0],[7,5],[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 0

        # condition for [o][s]
        l=[[2,2],[2,1]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[5][0] < pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l=[[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
        pl=[ch1,ch2]
        if pl in l:
            if (pts[0][0]>pts[8][0] and pts[0][0]>pts[4][0] and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and pts[5][0] > pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6,0],[6,6],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[8],pts[16]) < 52:
                ch1 = 2

        # condition for [gh][bdfikruvw]
        l = [[1,4],[1,5],[1,6],[1,3],[1,0]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1]<pts[20][1] and pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]:
                ch1 = 3

        # con for [gh][l]
        l=[[4,6],[4,1],[4,5],[4,3],[4,7]]
        pl=[ch1,ch2]
        if pl in l:
            if pts[4][0]>pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3],[5,0],[5,7], [5, 4], [5, 2],[5,1],[5,5]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[2][1]+15<pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[4],pts[11])>55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6],[1,1]]
        pl = [ch1, ch2]
        if pl in l:
            if (distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[4][0]<pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5],[2,4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[1][0] < pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6],[3,5],[3,4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]) and pts[4][1]>pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3,2],[3,1],[3,6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][1]+17>pts[8][1] and pts[4][1]+17>pts[12][1] and pts[4][1]+17>pts[16][1] and pts[4][1]+17>pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[4][0]>pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7],[5,2],[5,6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[3][0]<pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6],[4,2],[4,4],[4,1],[4,5],[4,7]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1] < pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[18][1] > pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0,4],[0,2],[0,3],[0,1],[0,6]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0]>pts[16][0]:
                ch1 = 6

        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[18][1] < pts[20][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1],[2,2],[2,6],[2,7],[2,0]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[8],pts[16])>50:
                ch1 = 6

        # con for [l][x]
        l = [[4, 6],[4,2],[4,1],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if distance(pts[4], pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1,4],[1,6],[1,0],[1,2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0] - pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7, 6],[4,6],[4,1],[4,2],[0, 2],[7, 1],[7, 4],[6, 6],[7, 2],[7, 5],[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                            pts[18][1] > pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0],[4,2],[4,1],[4,6],[4,4]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                    pts[18][1] > pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        l = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                            pts[18][1] < pts[20][1]) and (pts[2][0]<pts[0][0]) and pts[4][1]>pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2],[4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (distance(pts[4], pts[11]) < 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5],[3,6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                 pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1]<pts[4][1]):
                ch1 = 1

        l = [[6, 6],[6, 4], [6, 1],[6,2]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[5][0]-pts[4][0]-15<0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7, 5],[7, 1],[7, 6],[7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                         pts[18][1] > pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]
        pl = [ch1, ch2]
        if pl in l:
            if (pts[4][0]<pts[5][0]+15) and ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                         pts[18][1] > pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and
                 pts[18][1] < pts[20][1])) and pts[4][1]>pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]
        pl = [ch1, ch2]
        if pl in l:
            if not(pts[0][0]+fg < pts[8][0] and pts[0][0]+fg < pts[12][0] and pts[0][0]+fg < pts[16][0] and pts[0][0]+fg < pts[20][0]) and not(pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
                ch1 = 1

        l = [[5, 0], [5, 5],[0,1]]
        pl = [ch1, ch2]
        if pl in l:
            if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1]:
                ch1 = 1

        # ---- Subgroup classification (within-group letter determination) ----
        # [0->AEMNST][1->BFDIUVWKR][2->CO][3->GH][4->L][5->PQZ][6->X][7->YJ]

        if ch1 == 0:
            ch1 = 'S'
            if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                ch1 = 'A'
            if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                ch1 = 'T'
            if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                ch1 = 'E'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                ch1 = 'M'
            if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if distance(pts[12], pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (distance(pts[8], pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if distance(pts[8], pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                if pts[8][1] < pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] >pts[20][1]):
                ch1 = 'B'
            if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]):
                ch1 = 'D'
            if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 'F'
            if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 'I'
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 'W'
            if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1]<pts[9][1]:
                ch1 = 'K'
            if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 'U'
            if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] >pts[9][1]):
                ch1 = 'V'
            if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                ch1 = 'R'

        # Space gesture
        if ch1 == 1 or ch1 in ['E', 'S', 'X', 'Y', 'B']:
            if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                ch1 = 'Space'

        # Next gesture
        if ch1 in ['E', 'Y', 'B', 'Next']:
            if (pts[4][0] < pts[5][0]):
                ch1 = 'Next'

        # Backspace gesture
        if ch1 in ['Next', 'B', 'C', 'H', 'F', 'Backspace']:
            if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[4][1]<pts[8][1] and pts[4][1]<pts[12][1] and pts[4][1]<pts[16][1] and pts[4][1]<pts[20][1]:
                ch1 = 'Backspace'

        return ch1

    def inference_thread(self):
        while self.running:
            if self.frame is not None and self.model is not None:
                try:
                    img = cv2.flip(self.frame, 1)
                    hands, _ = self.hd.findHands(img, draw=False)
                    
                    if hands:
                        hand = hands[0]
                        x, y, w, h = hand['bbox']
                        img_h, img_w, _ = img.shape
                        y1, y2 = max(0, y - OFFSET), min(img_h, y + h + OFFSET)
                        x1, x2 = max(0, x - OFFSET), min(img_w, x + w + OFFSET)
                        crop = img[y1:y2, x1:x2]

                        if crop.size > 0:
                            # FIX D: Reset white canvas fresh each frame
                            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
                            handz, _ = self.hd2.findHands(crop, draw=False)
                            if handz:
                                pts = handz[0]['lmList']
                                ox, oy = ((400 - w) // 2) - 15, ((400 - h) // 2) - 15
                                
                                # Draw skeleton on white canvas (same as original)
                                for t in range(0, 4, 1):
                                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                                for t in range(5, 8, 1):
                                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                                for t in range(9, 12, 1):
                                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                                for t in range(13, 16, 1):
                                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                                for t in range(17, 20, 1):
                                    cv2.line(white, (pts[t][0]+ox, pts[t][1]+oy), (pts[t+1][0]+ox, pts[t+1][1]+oy), (0, 255, 0), 3)
                                cv2.line(white, (pts[5][0]+ox, pts[5][1]+oy), (pts[9][0]+ox, pts[9][1]+oy), (0, 255, 0), 3)
                                cv2.line(white, (pts[9][0]+ox, pts[9][1]+oy), (pts[13][0]+ox, pts[13][1]+oy), (0, 255, 0), 3)
                                cv2.line(white, (pts[13][0]+ox, pts[13][1]+oy), (pts[17][0]+ox, pts[17][1]+oy), (0, 255, 0), 3)
                                cv2.line(white, (pts[0][0]+ox, pts[0][1]+oy), (pts[5][0]+ox, pts[5][1]+oy), (0, 255, 0), 3)
                                cv2.line(white, (pts[0][0]+ox, pts[0][1]+oy), (pts[17][0]+ox, pts[17][1]+oy), (0, 255, 0), 3)
                                for i in range(21):
                                    cv2.circle(white, (pts[i][0]+ox, pts[i][1]+oy), 2, (0, 0, 255), 1)
                                
                                self.processed_img = white.copy()
                                
                                # ===== PREDICTION (DUAL MODE) =====
                                if self.current_mode == "ALPHABETS" and self.model is not None:
                                    img_input = white.reshape(1, 400, 400, 3)
                                    prob = np.array(self.model.predict(img_input, verbose=0)[0], dtype='float32')
                                    
                                    ch1 = np.argmax(prob)
                                    prob_copy = prob.copy()
                                    prob_copy[ch1] = 0
                                    ch2 = np.argmax(prob_copy)
                                    
                                    confidence = np.max(prob)
                                    char = self.subgroup_classify(ch1, ch2, pts)
                                    
                                elif self.current_mode == "NUMBERS" and self.numbers_model is not None:
                                    img_input = cv2.resize(white, (128, 128))
                                    img_input = img_input.reshape(1, 128, 128, 3) / 255.0
                                    prob = np.array(self.numbers_model.predict(img_input, verbose=0)[0], dtype='float32')
                                    
                                    num_idx = np.argmax(prob)
                                    confidence = np.max(prob)
                                    
                                    # Output the string version of the predicted number
                                    # We used classes 1 to 10 in training
                                    number_classes = [str(i) for i in range(1, 11)]
                                    char = number_classes[num_idx]
                                    
                                else:
                                    confidence = 0.0
                                    char = "..."
                                    
                                # Debug overlay
                                self.frame_count += 1
                                debug_text = f"Mode:{self.current_mode} | Conf:{confidence:.2f} | Res:{char}"
                                if self.frame_count % 30 == 0:
                                    print(f"[DEBUG] {debug_text}")
                                
                                if confidence > 0.75 and isinstance(char, str) and char != "...":
                                    self.current_symbol = char
                                    self.process_char_logic(char)
                                    
                                    # Speak
                                    if char != self.last_spoken and time.time() - self.last_spoken_time > 2.0:
                                        self.speak_async(char)
                                        self.last_spoken = char
                                        self.last_spoken_time = time.time()
                                else:
                                    self.current_symbol = "..."
                    else:
                        self.current_symbol = "..."
                except Exception as e:
                    print(f"Inference Error: {e}")
            time.sleep(0.05)

    def process_char_logic(self, char):
        if char == self.prev_char:
            return # Wait for change
        
        # Simple string builder
        if char == "Backspace":
            self.sentence = self.sentence[:-1]
        elif char == "Space":
            self.sentence += " "
        elif char == "Next":
            pass  # Skip 'Next' from being appended
        else:
            self.sentence += char
        
        self.prev_char = char
        
        # Update word suggestions
        st = self.sentence.rfind(" ")
        word = self.sentence[st+1:].strip()
        if HAS_ENCHANT and word:
            sugs = ddd.suggest(word)
            self.word1 = sugs[0] if len(sugs) > 0 else " "
            self.word2 = sugs[1] if len(sugs) > 1 else " "
            self.word3 = sugs[2] if len(sugs) > 2 else " "
            self.word4 = sugs[3] if len(sugs) > 3 else " "

    def video_loop(self):
        if self.frame is not None:
            display_frame = cv2.flip(self.frame.copy(), 1)
            
            # Debug overlay on camera feed (Task 4)
            debug_text = f"Sign: {self.current_symbol}"
            cv2.putText(display_frame, debug_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.panel.imgtk = img_tk
            self.panel.config(image=img_tk)
            
            self.lbl_char.config(text=self.current_symbol)
            self.lbl_sentence.config(text=self.sentence)
            for i, w in enumerate([self.word1, self.word2, self.word3, self.word4]):
                self.btn_sugs[i].config(text=w)
            
            img_tk2 = ImageTk.PhotoImage(image=Image.fromarray(self.processed_img))
            self.panel2.imgtk = img_tk2
            self.panel2.config(image=img_tk2)
            
        self.root.after(30, self.video_loop)

    def speak_async(self, text):
        import pyttsx3
        def _task():
            try:
                e = pyttsx3.init()
                e.say(text)
                e.runAndWait()
            except: pass
        threading.Thread(target=_task, daemon=True).start()

    def apply_sug(self, idx):
        words = [self.word1, self.word2, self.word3, self.word4]
        sug = words[idx]
        if sug.strip():
            start = self.sentence.rfind(" ")
            self.sentence = self.sentence[:start+1] + sug.upper() + " "
            self.speak_async(sug)

    def clear_fun(self):
        self.sentence = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "

    def destructor(self):
        self.running = False
        if self.vs: self.vs.release()
        self.root.destroy()

if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
