import os
import sys

# -------------------------------------------------------
# Add Project Root to Python Path
# -------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------
# Imports
# -------------------------------------------------------

import streamlit as st
from PIL import Image
from datetime import datetime

from backend.predictor import (
    predict_brain,
    predict_pneumonia
)

from backend.explainability import (
    generate_gradcam
)

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------



st.title("🔍 Explainable AI (XAI)")

st.markdown("""
Understand why NeuroVisionX made its prediction.

For image models, Grad-CAM highlights the important regions used by the neural network.
""")

st.divider()

# -------------------------------------------------------
# Model Selection
# -------------------------------------------------------
model = st.selectbox(
    "Select Model",
    [
        "Brain Tumor",
        "Pneumonia"
    ]
)

# -------------------------------------------------------
# File Upload
# -------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

run = st.button(
    "Generate Explainability",
    use_container_width=True
)

# -------------------------------------------------------
# Prediction & Explainability
# -------------------------------------------------------
if run:
    if uploaded_file is None:
        st.error("Please upload an image.")
        st.stop()

    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save image safely with timestamp
    os.makedirs("temp", exist_ok=True)
    image_path = os.path.join(
        "temp",
        f"{int(datetime.now().timestamp())}_{uploaded_file.name}"
    )
    image.save(image_path)

    # Run prediction + Grad-CAM
    if model == "Brain Tumor":
        prediction = predict_brain(image_path)
        gradcam = generate_gradcam(image_path, "brain")
    else:
        prediction = predict_pneumonia(image_path)
        gradcam = generate_gradcam(image_path, "pneumonia")

    st.divider()

    # Show prediction results
    st.subheader("Prediction")
    st.metric("Prediction", prediction["prediction"])
    st.metric("Confidence", f"{prediction['confidence']:.2f}%")

    st.divider()

    # Show Grad-CAM heatmap
    st.subheader("Grad-CAM Heatmap")
    st.image(gradcam, caption="Important Regions", use_container_width=True)

    st.success("Explainability Generated Successfully")
