import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), "AtoZ_3.1")
IMG_SIZE = (128, 128) # 128x128 is perfect for skeletons
BATCH_SIZE = 32
EPOCHS = 40

def train_pro_model():
    print("=== PRO SIGN AI TRAINER (Ultra-Accurate Mode) ===")
    
    # 1. Heavy Augmentation for confusion groups
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2, 
        rotation_range=20, 
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='constant',
        cval=255 
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, 
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, 
        class_mode='categorical', subset='validation'
    )

    if train_gen.num_classes < 2:
        print("ERROR: Not enough data folders! Collect data first.")
        return

    # 2. Advanced Multi-Class CNN Architecture
    model = Sequential([
        # First Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Second Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),
        
        # Third Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Transition/Classifier
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Training on {train_gen.num_classes} classes...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
    
    # 3. Save Model and Labels
    # Use the name the user's scripts expect
    model.save('landmark_cnn_model.h5')
    
    # Save indices to labels.pickle for the API
    indices = {v: k for k, v in train_gen.class_indices.items()}
    with open('labels.pickle', 'wb') as f:
        pickle.dump(indices, f)
        
    print("Done! Model and Labels saved successfully.")

if __name__ == "__main__":
    train_pro_model()
