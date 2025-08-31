# download_datasets.py

import os
import kaggle

def download_datasets():
    """Download all required datasets from Kaggle"""

    print("Downloading datasets...")

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Dataset 1: Crop pest and disease detection
    print("1. Downloading crop pest dataset...")
    print("   Source: https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection")
    kaggle.api.dataset_download_files(
        'nirmalsankalana/crop-pest-and-disease-detection',
        path='data/crop_pest',
        unzip=True
    )

    # Dataset 2: Agricultural Pests Dataset
    print("2. Downloading agricultural pests dataset...")
    print("   Source: https://www.kaggle.com/datasets/gauravduttakiit/agricultural-pests-dataset")
    kaggle.api.dataset_download_files(
        'gauravduttakiit/agricultural-pests-dataset',
        path='data/ag_pests',
        unzip=True
    )

    # Dataset 3: Pestopia Dataset
    print("3. Downloading pestopia dataset...")
    print("   Source: https://www.kaggle.com/datasets/shruthisindhura/pestopia")
    kaggle.api.dataset_download_files(
        'shruthisindhura/pestopia',
        path='data/pestopia',
        unzip=True
    )

    print("✅ All datasets downloaded successfully!")


if __name__ == "__main__":
    download_datasets()


