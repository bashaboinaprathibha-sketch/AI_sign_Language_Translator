"""
AI Sign Language Translator - Model Training Script
Trains a Convolutional Neural Network (CNN) on hand gesture images
and exports to both sign_language_model.keras and model.h5 formats.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "dataset/train"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def train_cnn():
    if not os.path.exists(DATASET_PATH):
        print(f"[Error] Dataset directory '{DATASET_PATH}' does not exist.")
        return

    print("==================================================")
    print(" AI Sign Language Recognition - Model Training")
    print("==================================================")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Image dimensions: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # Data Augmentation & Normalization
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )

    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = train_generator.num_classes
    print(f"\n[Info] Classes found ({num_classes}): {list(train_generator.class_indices.keys())}")

    # Build CNN Architecture
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train Model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # Save models in both .keras and .h5 format
    keras_path = "sign_language_model.keras"
    h5_path = "model.h5"

    model.save(keras_path)
    model.save(h5_path)

    print("\n✅ Model training complete!")
    print(f"✅ Saved model to: {keras_path}")
    print(f"✅ Saved model to: {h5_path}")

if __name__ == "__main__":
    train_cnn()
