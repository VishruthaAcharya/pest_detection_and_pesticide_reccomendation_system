# pest_detector.py

import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PestDetector:
    def __init__(self, model_path="models/pest_detection_model.h5", mapping_path="processed_data/class_mapping.json"):
        # Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = load_model(model_path)

        # Load class mapping
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Class mapping not found at {mapping_path}")
        with open(mapping_path, "r") as f:
            self.class_mapping = json.load(f)

        # Reverse mapping: index → pest name
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}

        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Loaded {len(self.class_mapping)} pest classes")

    def preprocess_image(self, img_path, target_size=(224, 224)):
        """Load and preprocess image for prediction"""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        return img_array

    def predict_pest(self, img_path):
        """Predict pest class from image"""
        processed_img = self.preprocess_image(img_path)
        predictions = self.model.predict(processed_img, verbose=0)[0]

        # Get best prediction
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        pest_name = self.idx_to_class.get(class_idx, "Unknown")

        return pest_name, float(confidence)

    def visualize_prediction(self, img_path, pest_name, confidence):
        """Draw prediction result on image"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Unable to read image: {img_path}")

        text = f"{pest_name} ({confidence:.2f})"
        cv2.putText(img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pest Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PestDetector()

    # Example test image (change this path as needed)
    test_image = "test_samples/test1.jpg"

    pest, conf = detector.predict_pest(test_image)
    print(f"🔍 Detected: {pest} with {conf:.2f} confidence")

    # Visualize result
    detector.visualize_prediction(test_image, pest, conf)
