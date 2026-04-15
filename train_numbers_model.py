import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Config
DATA_DIR = 'Numbers_Data'
MODEL_NAME = 'numbers_model.h5'
IMG_SIZE = (128, 128) # Smaller, faster model for numbers
BATCH_SIZE = 32
EPOCHS = 20

if not os.path.exists(DATA_DIR):
    print(f"ERROR: No '{DATA_DIR}' folder found. Please run data_collection_numbers.py first.")
    exit()

# Data Loading with /255 normalization and AUGMENTATION
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    rotation_range=15,      # Allow hand to be slightly tilted
    width_shift_range=0.1,  # Allow hand to be off-center
    height_shift_range=0.1,
    zoom_range=0.1,         # Allow hand to be closer/further
    horizontal_flip=False   # DO NOT flip (1 vs other numbers can change meaning)
)

# Flow from directory
# Note: classes order might be sorted alphabetically (1, 10, 2, 3...) 
# We explicitly define classes to keep 1 to 10 ordering
number_classes = [str(i) for i in range(1, 11)]

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=number_classes,
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=number_classes,
    subset='validation'
)

# Model Architecture - Simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') # 10 classes for 1-10
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
print("Starting training on numbers 1-10...")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save
model.save(MODEL_NAME)
print(f"Numbers Model saved successfully as {MODEL_NAME}")

# Also save class indices
import pickle
with open('numbers_labels.pickle', 'wb') as f:
    pickle.dump(train_generator.class_indices, f)
print("Labels saved to numbers_labels.pickle")
