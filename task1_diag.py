import tensorflow as tf
import numpy as np
import cv2
import os

model_path = "cnn8grps_rad1_model.h5"
image_path = "white.jpg"

if not os.path.exists(model_path):
    print(f"Error: {model_path} not found")
    exit()

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found")
    exit()

model = tf.keras.models.load_model(model_path)
print("=== MODEL INFO ===")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
model.summary()

# Test with blank white image
white = cv2.imread(image_path)
print("white.jpg shape:", white.shape)

# Test prediction on blank image
img = cv2.resize(white, (model.input_shape[1], model.input_shape[2]))
img = img.reshape(1, model.input_shape[1], model.input_shape[2], 3) / 255.0
pred = model.predict(img, verbose=0)
print("Blank image prediction:", np.round(pred, 4))
print("Predicted class index  :", np.argmax(pred))
