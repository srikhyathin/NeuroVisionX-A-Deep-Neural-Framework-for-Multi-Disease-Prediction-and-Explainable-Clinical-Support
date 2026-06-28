import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.title("🧠 About NeuroVisionX")

st.markdown("""
## AI Powered Multi-Disease Prediction System

**NeuroVisionX** is a deep learning and machine learning based healthcare platform
that predicts multiple diseases from symptoms, clinical parameters and medical images.

The project combines neural networks, image classification, and tabular data
analysis into a single intelligent healthcare application.
""")

st.divider()

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

st.header("🚀 Features")

st.markdown("""
- 🩺 Symptom Based Disease Prediction
- ❤️ Heart Disease Prediction
- 🩸 Diabetes Prediction
- 🧠 Brain Tumor Classification
- 🫁 Pneumonia Detection
- 📄 Automatic PDF Report Generation
- 💾 SQLite Database Storage
- 📊 Analytics Dashboard
- 📜 Patient History
""")

st.divider()

# ---------------------------------------------------------
# AI Models
# ---------------------------------------------------------

st.header("🤖 AI Models Used")

st.table({

    "Module":[

        "Symptoms",

        "Heart Disease",

        "Diabetes",

        "Brain Tumor",

        "Pneumonia"

    ],

    "Model":[

        "Feed Forward Neural Network",

        "Feed Forward Neural Network",

        "Feed Forward Neural Network",

        "EfficientNet-B0",

        "EfficientNet-B0"

    ]

})

st.divider()

# ---------------------------------------------------------
# Technologies
# ---------------------------------------------------------

st.header("💻 Technologies")

col1,col2,col3=st.columns(3)

with col1:

    st.markdown("""
### Frontend

- Streamlit
- Python
- Pandas
""")

with col2:

    st.markdown("""
### Backend

- PyTorch
- SQLite
- Joblib
""")

with col3:

    st.markdown("""
### AI

- Neural Networks
- EfficientNet
- Deep Learning
""")

st.divider()

# ---------------------------------------------------------
# Workflow
# ---------------------------------------------------------

st.header("⚙️ Workflow")

st.markdown("""

Patient Information

⬇️

Input Symptoms / Medical Parameters / Image

⬇️

Neural Network Prediction

⬇️

Confidence Score

⬇️

Disease Description

⬇️

Recommended Precautions

⬇️

PDF Report Generation

⬇️

Database Storage

""")

st.divider()

# ---------------------------------------------------------
# Developer
# ---------------------------------------------------------

st.header("👨‍💻 Developer")

st.success("""

NeuroVisionX

Developed as a Deep Learning based Medical Decision Support System.

""")

st.divider()

st.info("Version 1.0 | Built with ❤️ using Streamlit & PyTorch")