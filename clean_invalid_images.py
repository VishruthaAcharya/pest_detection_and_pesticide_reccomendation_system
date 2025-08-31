import os
from PIL import Image

# Folders to check
folders = [
    'processed_data/train',
    'processed_data/val',
]

def is_image_valid(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Check header
        # Try to fully load and resize the image to catch broken streams
        with Image.open(filepath) as img:
            img = img.convert('RGB')
            img.resize((10, 10))
        return True
    except Exception:
        return False

def clean_folder(folder):
    print(f'Checking {folder}...')
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            if not is_image_valid(path):
                print(f'Removing invalid image: {path}')
                os.remove(path)

if __name__ == '__main__':
    for folder in folders:
        clean_folder(folder)
    print('✅ Cleaning complete!')
