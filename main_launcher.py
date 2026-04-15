import tkinter as tk
import subprocess
import sys
import os

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Bidirectional Converter")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        # Center the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (400 // 2)
        y = (screen_height // 2) - (500 // 2)
        self.root.geometry(f'400x500+{x}+{y}')
        
        # UI
        main_frame = tk.Frame(root, padx=20, pady=40)
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        tk.Label(main_frame, text="ASL CONVERTER", font=("Arial", 20, "bold"), pady=10).pack()
        tk.Label(main_frame, text="Choose a mode:", font=("Arial", 14), pady=20).pack()
        
        btn_style = {
            "font": ("Arial", 12, "bold"),
            "pady": 15,
            "width": 25,
            "cursor": "hand2"
        }
        
        # Mode 1: Sign to Text
        self.btn1 = tk.Button(main_frame, text="\ud83d\udcf7 Sign \u2192 Text & Speech", 
                            bg="#3b82f6", fg="white", command=self.launch_sign_to_text, **btn_style)
        self.btn1.pack(pady=15)
        
        # Mode 2: Text to Sign
        self.btn2 = tk.Button(main_frame, text="\u2328\ufe0f Text \u2192 Sign Language", 
                            bg="#8b5cf6", fg="white", command=self.launch_text_to_sign, **btn_style)
        self.btn2.pack(pady=15)
        
        tk.Label(main_frame, text="Powered by CNN & MediaPipe", font=("Arial", 10), fg="gray", pady=40).pack(side=tk.BOTTOM)

    def launch_sign_to_text(self):
        script_path = os.path.join(os.path.dirname(__file__), "final_pred.py")
        subprocess.Popen([sys.executable, script_path])

    def launch_text_to_sign(self):
        script_path = os.path.join(os.path.dirname(__file__), "text_to_sign.py")
        subprocess.Popen([sys.executable, script_path])

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()
