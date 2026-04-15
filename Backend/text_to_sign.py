import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import os
import glob
import time

class TextToSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text \u2192 ASL Sign Language")
        self.root.geometry("600x750")
        self.root.resizable(False, False)
        
        self.running = False
        self.data_path = "AtoZ_3.1"
        
        # UI Layout
        # Row 1: Input
        input_frame = tk.Frame(root, pady=10)
        input_frame.pack()
        tk.Label(input_frame, text="Enter text:").pack(side=tk.LEFT)
        self.entry = tk.Entry(input_frame, width=30, font=("Arial", 12))
        self.entry.pack(side=tk.LEFT, padx=5)
        self.entry.bind("<Return>", lambda e: self.start_playback())
        
        # Row 2: Speed
        speed_frame = tk.Frame(root, pady=10)
        speed_frame.pack()
        tk.Label(speed_frame, text="Speed (ms/letter):").pack(side=tk.LEFT)
        self.speed_scale = tk.Scale(speed_frame, from_=300, to=3000, orient=tk.HORIZONTAL, length=200)
        self.speed_scale.set(1000)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Row 3: Buttons
        btn_frame = tk.Frame(root, pady=10)
        btn_frame.pack()
        self.play_btn = tk.Button(btn_frame, text="▶ Show Signs", bg="#22c55e", fg="white", 
                                font=("Arial", 12, "bold"), width=15, command=self.start_playback)
        self.play_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = tk.Button(btn_frame, text="\u23f9 Stop", bg="#ef4444", fg="white", 
                                font=("Arial", 12, "bold"), width=15, state=tk.DISABLED, command=self.stop_playback)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        # Center: Image Display
        self.img_label = tk.Label(root, width=400, height=400, bg="#1a1a2e")
        self.img_label.pack(pady=20)
        
        # Status Label
        self.status_label = tk.Label(root, text="Ready", font=("Arial", 14, "bold"))
        self.status_label.pack()
        
        # Progress Bar
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=20)

    def get_sign_image_path(self, letter):
        letter = letter.upper()
        folder = os.path.join(self.data_path, letter)
        if not os.path.isdir(folder):
            return None
        
        # Search for common image formats
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            files = glob.glob(os.path.join(folder, ext))
            if files:
                return files[0]
        return None

    def start_playback(self):
        text = self.entry.get().strip()
        if not text:
            messagebox.showwarning("Empty Input", "Please enter some text first.")
            return
            
        # Filter input to alpha and spaces
        self.text_to_show = "".join([c for c in text if c.isalpha() or c.isspace()])
        if not self.text_to_show:
            messagebox.showwarning("Invalid Input", "Text must contain letters or spaces.")
            return

        self.running = True
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Starting...")
        
        threading.Thread(target=self.playback_thread, daemon=True).start()

    def stop_playback(self):
        self.running = False

    def playback_thread(self):
        chars = list(self.text_to_show)
        n = len(chars)
        
        for i, char in enumerate(chars):
            if not self.running:
                break
                
            self.root.after(0, self.update_progress, i + 1, n)
            
            if char.isspace():
                self.root.after(0, self.show_space)
            else:
                img_path = self.get_sign_image_path(char)
                if img_path:
                    self.root.after(0, self.show_image, img_path, char)
                else:
                    self.root.after(0, self.show_missing, char)
            
            # Wait for the specified speed
            time.sleep(self.speed_scale.get() / 1000.0)
            
        self.root.after(0, self.reset_ui)

    def update_progress(self, current, total):
        val = (current / total) * 100
        self.progress['value'] = val

    def show_image(self, path, char):
        try:
            img = Image.open(path)
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text="")
            self.img_label._photo = photo # Reference to prevent GC
            self.status_label.config(text=f"Letter: {char}")
        except Exception as e:
            print(f"Error loading image: {e}")

    def show_space(self):
        self.img_label.config(image="", text="[ SPACE ]", fg="white", font=("Arial", 24, "bold"))
        self.status_label.config(text="Space")

    def show_missing(self, char):
        self.img_label.config(image="", text=f"No sign for: {char}", fg="yellow", font=("Arial", 18))
        self.status_label.config(text=f"Missing: {char}")

    def reset_ui(self):
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Finished")
        self.running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = TextToSignApp(root)
    root.mainloop()
