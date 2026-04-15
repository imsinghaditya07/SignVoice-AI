import os
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialization
detector = HandDetector(maxHands=1)
DATA_DIR = 'AtoZ_3.1'
data = []
labels = []

print("Extracting landmarks from images... this may take a minute.")

# Loop through each folder (A-Z)
for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue
    
    print(f"Processing {label}...")
    # Limit images for quick training (100 per class is plenty for landmarks)
    count = 0
    for img_name in os.listdir(label_path):
        if count >= 100: break
        
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # We need to find hands in these pre-drawn skeleton images
        # BUT wait, the images ARE landmarks! 
        # Actually, it's easier to just detect from the raw images if they were raw.
        # Since these are skeleton images, I will extract the coordinates from the green lines if possible
        # OR better: I will just use the CNN approach but with a smaller architecture since it's just green lines.
        
        # Actually, let's stick to the CNN approach but optimize it for speed.
        pass

# --- REVISED PLAN: Fast CNN training on the existing skeletons ---
# Because the dataset images are already skeletons, we don't need to 'detect' them again.
# We just need a model that understands these green line patterns.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train_fast_cnn():
    IMG_SIZE = (128, 128) # Smaller for speed, plenty for simple skeletons
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=32, class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=32, class_mode='categorical', subset='validation'
    )

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training fast model...")
    model.fit(train_gen, epochs=5, validation_data=val_gen)
    
    model.save('landmark_cnn_model.h5')
    
    # Save labels
    with open('labels.pickle', 'wb') as f:
        pickle.dump(train_gen.class_indices, f)

if __name__ == "__main__":
    train_fast_cnn()
