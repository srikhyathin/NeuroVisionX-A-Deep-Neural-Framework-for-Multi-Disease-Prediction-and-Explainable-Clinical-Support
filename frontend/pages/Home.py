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

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------



# -------------------------------------------------------
# Home Page
# -------------------------------------------------------

st.title("🧠 NeuroVisionX")

st.subheader(
    "Multi-Modal Neural Intelligence for Early Disease Prediction"
)

st.markdown("---")
col1, col2, col3 = st.columns(3)

col1.metric("Diseases", "40+")
col2.metric("AI Models", "5")
col3.metric("Explainability", "SHAP + GradCAM")

st.markdown("---")

st.header("🚀 Features")

st.write("""
✅ Symptom-Based Disease Prediction

✅ Diabetes Prediction

✅ Heart Disease Prediction

✅ Brain Tumor Detection

✅ Pneumonia Detection

✅ Explainable AI

✅ Disease Knowledge Graph

✅ Patient Report Generation

✅ Analytics Dashboard

✅ SQLite Database
""")

st.markdown("---")

st.header("🧠 AI Models")

st.info("""
• Feed Forward Neural Network

• EfficientNet-B0

• NeuroFusion Engine

• SHAP Explainability

• Grad-CAM
""")

st.markdown("---")

st.header("🛠 Technology Stack")

st.write("""
Python

PyTorch

FastAPI

Streamlit

SQLite

OpenCV

SHAP

NetworkX

ReportLab
""")

st.markdown("---")

st.success(
    "Welcome to NeuroVisionX! Select a module from the left sidebar to begin."
)