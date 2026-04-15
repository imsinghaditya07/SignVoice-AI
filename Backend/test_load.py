import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import cv2
from keras.models import load_model

try:
    model = load_model('cnn8grps_rad1_model.h5')
    print("SUCCESS: Model loaded successfully.")
except Exception as e:
    print(f"FAILURE: Could not load model. Error: {e}")
