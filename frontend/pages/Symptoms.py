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

import pandas as pd
import streamlit as st

from backend.predictor import predict_symptoms
from backend.database import NeuroVisionDatabase
from backend.report_generator import ReportGenerator
from backend.disease_info import DISEASE_INFO

# -------------------------------------------------------
# Initialize Backend
# -------------------------------------------------------

db = NeuroVisionDatabase()

report_generator = ReportGenerator()

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------


st.title("🩺 Symptom Based Disease Prediction")

st.markdown(
"""
Select the symptoms experienced by the patient.

NeuroVisionX AI will predict the most probable disease
based on the selected symptoms.
"""
)

st.divider()

# ----------------------------------------------------
# Patient Details
# ----------------------------------------------------

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    patient_name = st.text_input(
        "Patient Name"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

with col2:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    phone = st.text_input(
        "Phone (Optional)"
    )

st.divider()

# ----------------------------------------------------
# Load Symptoms
# ----------------------------------------------------

st.subheader("🦠 Select Symptoms")

try:

    df = pd.read_csv(
        "data/symptoms/processed_symptoms.csv"
    )

except Exception:

    st.error(
        "processed_symptoms.csv not found."
    )

    st.stop()

symptoms = list(df.columns[:-1])

selected = []

selected_symptoms = []

cols = st.columns(3)

for i, symptom in enumerate(symptoms):

    with cols[i % 3]:

        checked = st.checkbox(
            symptom.replace("_", " ").title(),
            key=symptom
        )

        if checked:

            selected.append(1)

            selected_symptoms.append(
                symptom.replace("_", " ").title()
            )

        else:

            selected.append(0)

predict = st.button(

    "🔍 Predict Disease",

    use_container_width=True
)
# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if predict:

    if patient_name.strip() == "":

        st.error("Please enter the patient name.")

        st.stop()

    if len(selected_symptoms) == 0:

        st.warning("Please select at least one symptom.")

        st.stop()

    with st.spinner("Predicting Disease..."):

        result = predict_symptoms(selected)

    if not result["success"]:

        st.error(result["error"])

        st.stop()

    prediction = result["prediction"]

    confidence = result["confidence"]

    info = DISEASE_INFO.get(

        prediction,

        {

            "description": "Description unavailable.",

            "precautions": [

                "Consult your doctor."

            ]

        }

    )

    # ---------------------------------------------
    # Generate PDF Report
    # ---------------------------------------------

    pdf_path = report_generator.generate_report(

        patient_name=patient_name,

        age=age,

        gender=gender,

        prediction=prediction,

        confidence=confidence,

        symptoms=selected_symptoms,

        description=info["description"],

        precautions=info["precautions"],

        model_used="SymptomNet"

    )

    # ---------------------------------------------
    # Save Prediction
    # ---------------------------------------------

    db.save_prediction(

        patient_name=patient_name,

        age=age,

        gender=gender,

        disease=prediction,

        confidence=confidence,

        model_used="SymptomNet",

        symptoms=selected_symptoms,

        report_path=pdf_path

    )

    # ---------------------------------------------
    # Display Prediction
    # ---------------------------------------------

    st.success("Prediction Completed Successfully")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Predicted Disease",

            prediction

        )

    with col2:

        st.metric(

            "Confidence",

            f"{confidence:.2f}%"

        )

    if confidence >= 90:

        st.success("🟢 High Confidence Prediction")

    elif confidence >= 70:

        st.warning("🟡 Medium Confidence Prediction")

    else:

        st.error("🔴 Low Confidence Prediction")
            # ----------------------------------------------------
    # Disease Information
    # ----------------------------------------------------

    st.divider()

    st.subheader("📖 Disease Description")

    st.info(info["description"])

    st.subheader("🛡 Recommended Precautions")

    for precaution in info["precautions"]:

        st.write(f"✅ {precaution}")

    # ----------------------------------------------------
    # Symptoms Selected
    # ----------------------------------------------------

    st.divider()

    st.subheader("🦠 Symptoms Selected")

    if len(selected_symptoms) > 0:

        for symptom in selected_symptoms:

            st.write(f"• {symptom}")

    else:

        st.write("No symptoms selected.")

    # ----------------------------------------------------
    # Download Report
    # ----------------------------------------------------

    st.divider()

    st.subheader("📄 Medical Report")

    try:

        with open(pdf_path, "rb") as pdf_file:

            st.download_button(

                label="📥 Download PDF Report",

                data=pdf_file,

                file_name=os.path.basename(pdf_path),

                mime="application/pdf",

                use_container_width=True

            )

    except Exception:

        st.warning("Unable to load generated PDF report.")

    # ----------------------------------------------------
    # Prediction Summary
    # ----------------------------------------------------

    st.divider()

    st.success("✅ NeuroVisionX Analysis Completed")

    st.markdown(f"""
### Summary

**Patient Name:** {patient_name}

**Age:** {age}

**Gender:** {gender}

**Predicted Disease:** **{prediction}**

**Confidence:** **{confidence:.2f}%**

**AI Model Used:** SymptomNet
""")
