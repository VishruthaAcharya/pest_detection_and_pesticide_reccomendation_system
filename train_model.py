# pest_detection_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import numpy as np
import matplotlib.pyplot as plt
import os


class PestDetectionModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None

        # Load class mapping
        with open("processed_data/class_mapping.json", "r") as f:
            self.class_mapping = json.load(f)

        self.num_classes = len(self.class_mapping)
        print(f"Training model for {self.num_classes} pest classes")

    def create_model(self):
        """Create CNN model for pest detection using transfer learning"""

        # Base model: MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False  # Freeze base layers

        # Custom classification head (diamond shape)
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax")
        ])

        # Compile model
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("✅ Model created successfully!")
        self.model.summary()

    def prepare_data(self):
        """Prepare training and validation data with augmentation"""

        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # Create data generators
        self.train_generator = train_datagen.flow_from_directory(
            "processed_data/train",
            target_size=self.input_shape[:2],
            batch_size=40,  # increased from 32
            class_mode="categorical"
        )

        self.val_generator = val_datagen.flow_from_directory(
            "processed_data/val",
            target_size=self.input_shape[:2],
            batch_size=40,  # increased from 32
            class_mode="categorical"
        )

        print("✅ Data generators created successfully!")

    def train_model(self, epochs=15):
        """Train the model"""

        print("🚀 Starting model training...")

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2, patience=2, min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "models/best_model.h5",
                save_best_only=True,
                monitor="val_accuracy"
            )
        ]

        os.makedirs("models", exist_ok=True)

        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks
        )

        # Save final model
        self.model.save("models/pest_detection_model.h5")
        print("✅ Model training completed!")

        self.plot_training_history()

    def plot_training_history(self):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        ax1.plot(self.history.history["accuracy"], label="Training Accuracy")
        ax1.plot(self.history.history["val_accuracy"], label="Validation Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # Loss
        ax2.plot(self.history.history["loss"], label="Training Loss")
        ax2.plot(self.history.history["val_loss"], label="Validation Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.show()

        print("📊 Training history saved as 'training_history.png'")


if __name__ == "__main__":
    print("[DEBUG] Script started.")
    model = PestDetectionModel()
    print("[DEBUG] Model class initialized.")
    model.create_model()
    print("[DEBUG] Model created.")
    model.prepare_data()
    print("[DEBUG] Data prepared.")
    model.train_model(epochs=15)
    print("[DEBUG] Training complete.")
