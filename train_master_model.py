import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# ======================================================================
# --- 1. CONFIGURATION ---
# ======================================================================
# CHOOSE WHICH DISEASE TO TRAIN A MODEL FOR.
# Valid options: 'PNEUMONIA', 'COVID19', 'TURBERCULOSIS'
DISEASE_TO_TRAIN = 'PNEUMONIA'  # <-- We will change this later for the other models

# The name of your dataset folder
BASE_DATA_DIR = 'X-Ray_Dataset' 

# Model training settings
EPOCHS = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# ======================================================================
# --- 2. SETUP PATHS (Automatic) ---
# ======================================================================
# This section automatically sets up the correct folders and file names
# based on the DISEASE_TO_TRAIN variable you set above.
train_dir = os.path.join(BASE_DATA_DIR, 'train')
val_dir = os.path.join(BASE_DATA_DIR, 'val')
test_dir = os.path.join(BASE_DATA_DIR, 'test')
SAVED_MODEL_NAME = f'{DISEASE_TO_TRAIN.lower()}_classifier.h5'

# ======================================================================
# --- 3. PREPARE THE DATA GENERATORS ---
# ======================================================================
# These generators will read images from the folders and prepare them for the model.
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# This is the key part: we tell the generator to ONLY look for images
# in the 'NORMAL' folder and the specific disease folder we chose.
class_list = ['NORMAL', DISEASE_TO_TRAIN]
print(f"Training a binary classifier for: {class_list}")

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    class_mode='binary', classes=class_list, shuffle=True
)
validation_generator = validation_datagen.flow_from_directory(
    val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    class_mode='binary', classes=class_list, shuffle=False
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    class_mode='binary', classes=class_list, shuffle=False
)

# ======================================================================
# --- 4. BUILD AND COMPILE THE CNN MODEL ---
# ======================================================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Helps prevent the model from just memorizing the data
    Dense(1, activation='sigmoid') # Sigmoid is perfect for 2-class (binary) problems
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary() # Print the model architecture

# ======================================================================
# --- 5. TRAIN, SAVE, AND EVALUATE THE MODEL ---
# ======================================================================
print(f"\n--- Starting training for {DISEASE_TO_TRAIN} ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)
print("--- Training finished! ---")

# Save the fully trained model
model.save(SAVED_MODEL_NAME)
print(f"Model saved successfully to '{SAVED_MODEL_NAME}'")

# Evaluate the model on the unseen test data
print("\n--- Evaluating model on the test set... ---")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Final Test Accuracy for {DISEASE_TO_TRAIN}: {test_accuracy * 100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")