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


from backend.predictor import predict_diabetes
from backend.database import NeuroVisionDatabase
from backend.report_generator import ReportGenerator
from backend.disease_info import DISEASE_INFO

# -------------------------------------------------------
# Initialize
# -------------------------------------------------------

db = NeuroVisionDatabase()

report_generator = ReportGenerator()

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------



st.title("🩸 Diabetes Prediction")

st.markdown("""

Predict Diabetes using the trained Neural Network model.

""")

st.divider()

# ---------------------------------------------------
# Patient Information
# ---------------------------------------------------

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    patient_name = st.text_input("Patient Name")

    patient_age = st.number_input(

        "Patient Age",

        1,

        120,

        30

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

# ---------------------------------------------------
# Diabetes Parameters
# ---------------------------------------------------

st.subheader("🩺 Diabetes Parameters")

c1, c2 = st.columns(2)

with c1:

    pregnancies = st.number_input(

        "Pregnancies",

        0,

        20,

        0

    )

    glucose = st.number_input(

        "Glucose",

        0,

        300,

        120

    )

    blood_pressure = st.number_input(

        "Blood Pressure",

        0,

        200,

        70

    )

    skin_thickness = st.number_input(

        "Skin Thickness",

        0,

        100,

        20

    )

with c2:

    insulin = st.number_input(

        "Insulin",

        0,

        900,

        80

    )

    bmi = st.number_input(

        "BMI",

        0.0,

        70.0,

        25.0

    )

    diabetes_pedigree = st.number_input(

        "Diabetes Pedigree Function",

        0.0,

        3.0,

        0.5

    )

    age = st.number_input(

        "Age",

        1,

        120,

        30

    )

predict = st.button(

    "🩸 Predict Diabetes",

    use_container_width=True

)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if predict:

    if patient_name.strip() == "":

        st.error("Please enter the patient name.")

        st.stop()

    input_data = [

        pregnancies,

        glucose,

        blood_pressure,

        skin_thickness,

        insulin,

        bmi,

        diabetes_pedigree,

        age

    ]

    with st.spinner("Predicting Diabetes..."):

        result = predict_diabetes(input_data)

    if not result["success"]:

        st.error(result["error"])

        st.stop()

    prediction = result["prediction"]

    confidence = result["confidence"]

    info = DISEASE_INFO.get(

        prediction,

        {

            "description":

            "Description unavailable.",

            "precautions":[

                "Consult an Endocrinologist."

            ]

        }

    )

    # ---------------------------------------------------
    # Generate PDF Report
    # ---------------------------------------------------

    pdf_path = report_generator.generate_report(

        patient_name=patient_name,

        age=patient_age,

        gender=gender,

        prediction=prediction,

        confidence=confidence,

        symptoms=[

            f"Pregnancies : {pregnancies}",

            f"Glucose : {glucose}",

            f"Blood Pressure : {blood_pressure}",

            f"Skin Thickness : {skin_thickness}",

            f"Insulin : {insulin}",

            f"BMI : {bmi}",

            f"Diabetes Pedigree : {diabetes_pedigree}",

            f"Age : {age}"

        ],

        description=info["description"],

        precautions=info["precautions"],

        model_used="DiabetesNet"

    )

    # ---------------------------------------------------
    # Save Prediction
    # ---------------------------------------------------

    db.save_prediction(

        patient_name=patient_name,

        age=patient_age,

        gender=gender,

        disease=prediction,

        confidence=confidence,

        model_used="DiabetesNet",

        symptoms=[

            str(pregnancies),

            str(glucose),

            str(blood_pressure),

            str(skin_thickness),

            str(insulin),

            str(bmi),

            str(diabetes_pedigree),

            str(age)

        ],

        report_path=pdf_path

    )

    # ---------------------------------------------------
    # Display Prediction
    # ---------------------------------------------------

    st.success("Prediction Completed Successfully")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(

            "Prediction",

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

        # ---------------------------------------------------
    # Disease Information
    # ---------------------------------------------------

    st.divider()

    st.subheader("📖 Disease Description")

    st.info(info["description"])

    st.subheader("🛡 Recommended Precautions")

    for precaution in info["precautions"]:

        st.write(f"✅ {precaution}")

    # ---------------------------------------------------
    # Diabetes Parameters Summary
    # ---------------------------------------------------

    st.divider()

    st.subheader("🩸 Diabetes Parameters Summary")

    summary = {

        "Pregnancies": pregnancies,

        "Glucose": glucose,

        "Blood Pressure": blood_pressure,

        "Skin Thickness": skin_thickness,

        "Insulin": insulin,

        "BMI": bmi,

        "Diabetes Pedigree Function": diabetes_pedigree,

        "Age": age

    }

    for key, value in summary.items():

        st.write(f"**{key}:** {value}")

    # ---------------------------------------------------
    # Download PDF Report
    # ---------------------------------------------------

    st.divider()

    st.subheader("📄 Medical Report")

    try:

        with open(pdf_path, "rb") as pdf_file:

            st.download_button(

                label="📥 Download Diabetes Report",

                data=pdf_file,

                file_name=os.path.basename(pdf_path),

                mime="application/pdf",

                use_container_width=True

            )

    except Exception:

        st.warning("Unable to load generated PDF report.")

    # ---------------------------------------------------
    # Final Summary
    # ---------------------------------------------------

    st.divider()

    st.success("✅ Diabetes Analysis Completed")

    st.markdown(f"""
### Patient Summary

**Patient Name:** {patient_name}

**Age:** {patient_age}

**Gender:** {gender}

**Predicted Result:** **{prediction}**

**Confidence:** **{confidence:.2f}%**

**AI Model Used:** DiabetesNet
""")