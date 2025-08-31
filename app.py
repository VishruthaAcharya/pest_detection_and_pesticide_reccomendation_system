# app.py

import streamlit as st
import numpy as np
from PIL import Image
import os

# Import our custom modules
from pest_detector import PestDetector
from pesticide_recommender import PesticideRecommender

# Configure Streamlit page
st.set_page_config(
    page_title="Pest Detection & Pesticide Recommendation System",
    page_icon="🐛",
    layout="wide"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        margin-bottom: 1rem;
    }
    .pest-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #4CAF50;
    }
    .pesticide-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #2196F3;
    }
    .no-pest-card {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #FF9800;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


class PestDetectionApp:
    def __init__(self):
        self.detector = None
        self.recommender = None
        self.initialize_components()

    def initialize_components(self):
        """Initialize detector and recommender"""
        try:
            self.detector = PestDetector()
            self.recommender = PesticideRecommender()
            return True
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return False

    def run(self):
        """Main application function"""
        # Header
        st.markdown(
            '<h1 class="main-header">🐛 Pest Detection & Pesticide Recommendation System</h1>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem;">
                <p style="font-size: 1.2rem; color: #666;">
                    Upload an image of a crop leaf or fruit to detect pests and get pesticide recommendations
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Check if components are initialized
        if self.detector is None or self.recommender is None:
            st.error("❌ System not properly initialized. Please check if you've trained the model first.")
            st.info("Run the training script: `python train_model.py`")
            return

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a crop leaf or fruit"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<h2 class="sub-header">📷 Uploaded Image</h2>', unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with col2:
                st.markdown('<h2 class="sub-header">🔍 Detection Results</h2>', unsafe_allow_html=True)

                if st.button("🚀 Detect Pest", type="primary"):
                    self.process_image(image)

            # Instructions
            with st.expander("📋 How to Use This System"):
                st.markdown(
                    """
                    1. **Upload an Image**: Select a clear photo of a crop leaf or fruit  
                    2. **Detect Pest**: Click the 'Detect Pest' button  
                    3. **View Results**: See detected pests with confidence scores  
                    4. **Get Recommendations**: View recommended pesticides and application rates  
                    5. **Take Action**: Use the recommendations to protect your crops  

                    **Tips for Better Results:**  
                    - Use clear, well-lit images  
                    - Focus on the affected area of the plant  
                    - Avoid blurry or dark images  
                    - Include the pest clearly in the frame  
                    """
                )

    def process_image(self, image):
        """Process uploaded image and display results"""
        with st.spinner("🔍 Analyzing image for pests..."):
            detection_result = self.detector.detect_pest(image, confidence_threshold=0.3)

        if detection_result["pest_detected"]:
            self.display_pest_detected(detection_result, image)
        else:
            self.display_no_pest_detected()

    def display_pest_detected(self, detection_result, original_image):
        """Display pest detection results"""
        primary_pest = detection_result["primary_pest"]
        confidence = detection_result["confidence"]

        st.markdown(
            f"""
            <div class="pest-card">
                <h3>🐛 Pest Detected!</h3>
                <p><strong>Primary Pest:</strong> {primary_pest.title()}</p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # All predictions
        st.markdown("### 🔎 All Predictions")
        for pred in detection_result["all_predictions"][:3]:
            emoji = "🟢" if pred["confidence"] > 0.7 else "🟡" if pred["confidence"] > 0.4 else "🔴"
            st.write(f"{emoji} **{pred['class'].title()}**: {pred['confidence']:.1%}")

        # Visualization
        st.markdown("### 📸 Detection Visualization")
        detected_image = self.detector.draw_detection_box(original_image, detection_result)
        st.image(detected_image, caption="Detected Pest Location", use_column_width=True)

        # Recommendations
        self.display_pesticide_recommendations(primary_pest)

    def display_no_pest_detected(self):
        """Display no pest detected message"""
        st.markdown(
            """
            <div class="no-pest-card">
                <h3>✅ No Pest Detected</h3>
                <p>The system didn't detect any pests in this image with sufficient confidence.</p>
                <p><em>Your crops appear to be healthy! 🌱</em></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            """
            **Tips if you expected to find a pest:**  
            - Try uploading a clearer image  
            - Ensure the pest is clearly visible  
            - Check if the lighting is adequate  
            - The pest might not be in our database yet  
            """
        )

    def display_pesticide_recommendations(self, pest_name):
        """Display pesticide recommendations for detected pest"""
        st.markdown("### 💊 Pesticide Recommendations")

        recommendations = self.recommender.get_recommendations(pest_name)

        if recommendations:
            st.success(f"Found {len(recommendations)} pesticide options for {pest_name.title()}")

            for i, rec in enumerate(recommendations, 1):
                emoji = {
                    "High": "🟢",
                    "Medium": "🟡",
                    "Low": "🔴"
                }.get(rec["effectiveness"], "⚪")

                st.markdown(
                    f"""
                    <div class="pesticide-card">
                        <h4>{i}. {rec['pesticide_name']}</h4>
                        <p><strong>Application Rate:</strong> {rec['application_rate']}</p>
                        <p><strong>Effectiveness:</strong> {emoji} {rec['effectiveness']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.warning(
                """
                ⚠ **Important Safety Notes:**  
                - Always read and follow pesticide label instructions  
                - Wear protective equipment when applying  
                - Follow pre-harvest interval guidelines  
                - Consult with agricultural experts if unsure  
                - Consider integrated pest management approaches  
                """
            )
        else:
            st.warning(f"No specific pesticide recommendations found for {pest_name.title()}")
            st.info("Consider consulting with a local agricultural expert for treatment options.")


# Main execution
if __name__ == "__main__":
    app = PestDetectionApp()
    app.run()
