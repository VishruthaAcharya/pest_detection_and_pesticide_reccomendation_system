# preprocess_data.py

import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import json


class DataPreprocessor:
    def __init__(self):
        self.pest_classes = {}  # Maps pest name -> index
        self.class_mapping = {}  # Reserved for reverse mapping if needed

    def create_unified_dataset(self):
        """Combine datasets and create unified pest classes"""
        print("Creating unified dataset...")

        # Create directories for processed data
        os.makedirs('processed_data/images', exist_ok=True)
        os.makedirs('processed_data/train', exist_ok=True)
        os.makedirs('processed_data/val', exist_ok=True)

        # Process crop pest dataset
        self.process_crop_pest_data()

        # Process agricultural pest dataset
        self.process_ag_pest_data()

        # Create train/validation split
        self.create_train_val_split()

        # Save class mapping
        self.save_class_mapping()

        print("✅ Data preprocessing completed!")

    def process_crop_pest_data(self):
        """Process the crop pest dataset"""
        crop_pest_path = 'data/crop_pest'
        print(f"Checking crop pest path: {crop_pest_path}")
        if os.path.exists(crop_pest_path):
            for root, dirs, files in os.walk(crop_pest_path):
                print(f"Walking: {root}, {len(files)} files")
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        pest_class = self.extract_pest_class(root)
                        print(f"Found file: {file}, pest_class: {pest_class}")
                        if pest_class:
                            self.copy_image_with_class(
                                os.path.join(root, file),
                                pest_class
                            )
        else:
            print(f"Path does not exist: {crop_pest_path}")

    def process_ag_pest_data(self):
        """Process the agricultural pest dataset"""
        ag_pest_path = 'data/ag_pests'
        print(f"Checking ag pest path: {ag_pest_path}")
        if os.path.exists(ag_pest_path):
            for root, dirs, files in os.walk(ag_pest_path):
                print(f"Walking: {root}, {len(files)} files")
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        pest_class = self.extract_pest_class(root)
                        print(f"Found file: {file}, pest_class: {pest_class}")
                        if pest_class:
                            self.copy_image_with_class(
                                os.path.join(root, file),
                                pest_class
                            )
        else:
            print(f"Path does not exist: {ag_pest_path}")

    def extract_pest_class(self, path):
        """Extract pest class name from file path"""
        pest_keywords = [
            'aphid', 'thrips', 'whitefly', 'caterpillar', 'beetle',
            'mite', 'leafhopper', 'scale', 'borer', 'weevil',
            'armyworm', 'bollworm', 'cutworm', 'wireworm'
        ]

        path_lower = path.lower()
        for keyword in pest_keywords:
            if keyword in path_lower:
                return keyword

        # If no keyword found, use folder name
        folder_name = os.path.basename(path).lower()
        if folder_name and folder_name != 'images':
            return folder_name

        return None

    def copy_image_with_class(self, src_path, pest_class):
        """Copy image to processed data with class label"""
        if pest_class not in self.pest_classes:
            self.pest_classes[pest_class] = len(self.pest_classes)

        # Create class directory
        class_dir = f'processed_data/images/{pest_class}'
        os.makedirs(class_dir, exist_ok=True)

        # Copy file with unique name
        filename = f"{pest_class}_{len(os.listdir(class_dir))}.jpg"
        dst_path = os.path.join(class_dir, filename)
        print(f"Copying {src_path} to {dst_path}")
        shutil.copy2(src_path, dst_path)

    def create_train_val_split(self):
        """Create training and validation splits"""
        for pest_class in self.pest_classes.keys():
            class_path = f'processed_data/images/{pest_class}'
            if os.path.exists(class_path):
                images = os.listdir(class_path)

                # Split 80% train, 20% validation
                train_images, val_images = train_test_split(
                    images, test_size=0.2, random_state=42
                )

                # Create class directories in train/val
                os.makedirs(f'processed_data/train/{pest_class}', exist_ok=True)
                os.makedirs(f'processed_data/val/{pest_class}', exist_ok=True)

                # Copy training images
                for img in train_images:
                    shutil.copy2(
                        os.path.join(class_path, img),
                        f'processed_data/train/{pest_class}/{img}'
                    )

                # Copy validation images
                for img in val_images:
                    shutil.copy2(
                        os.path.join(class_path, img),
                        f'processed_data/val/{pest_class}/{img}'
                    )

    def save_class_mapping(self):
        """Save class mapping for later use"""
        with open('processed_data/class_mapping.json', 'w') as f:
            json.dump(self.pest_classes, f, indent=4)

        print(f"📂 Found {len(self.pest_classes)} pest classes:")
        for pest, idx in self.pest_classes.items():
            print(f"  {idx}: {pest}")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.create_unified_dataset()
