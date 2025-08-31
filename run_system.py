import subprocess
import sys
import os
import time


def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'models/pest_detection_model.h5',
        'processed_data/class_mapping.json',
        'app.py'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    return missing_files


def setup_system():
    """Set up the entire system from scratch"""
    print("\n🚀 Setting up Pest Detection System...")

    steps = [
        ("📥 Downloading datasets...", "python download_datasets.py"),
        ("🔄 Preprocessing data...", "python preprocess_data.py"),
        ("🧠 Training model (this may take a while)...", "python train_model.py"),
        ("🧪 Testing pesticide recommender...", "python pesticide_recommender.py"),
    ]

    for description, command in steps:
        print(f"\n{description}")
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Success!")
            else:
                print(f"❌ Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error executing {command}: {e}")
            return False

    return True


def run_app():
    """Run the Streamlit app"""
    print("\n🌐 Starting Pest Detection Web App...")
    print("The app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")

    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped!")
    except Exception as e:
        print(f"❌ Error running app: {e}")


def main():
    print("\n🐛 Pest Detection and Pesticide Recommendation System")
    print("=" * 60)

    # Check if system is already set up
    missing_files = check_requirements()

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Setting up system from scratch...")

        if setup_system():
            print("\n✅ System setup completed successfully!")
        else:
            print("\n❌ System setup failed. Please check error messages above.")
            return
    else:
        print("✅ System already set up!")

    # Run the app
    run_app()


if __name__ == "__main__":
    main()
