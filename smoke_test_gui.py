import subprocess
import time
import sys
from PIL import ImageGrab
import os

def capture_app(script_name, output_name):
    print(f"Launching {script_name}...")
    proc = subprocess.Popen([sys.executable, script_name])
    time.sleep(5) # Wait for window to appear
    
    print(f"Capturing screen for {script_name}...")
    screenshot = ImageGrab.grab()
    screenshot.save(output_name)
    
    print(f"Closing {script_name}...")
    proc.terminate()
    time.sleep(1)

if __name__ == "__main__":
    # Create screenshots directory
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
        
    capture_app("main_launcher.py", "screenshots/launcher.png")
    capture_app("text_to_sign.py", "screenshots/text_to_sign.png")
    print("Smoke tests complete. Screenshots saved in screenshots/ folder.")
