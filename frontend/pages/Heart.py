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
from backend.predictor import predict_heart
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


st.title("❤️ Heart Disease Prediction")

st.markdown(

"""
Predict the probability of Heart Disease
using the trained Neural Network model.

"""

)

st.divider()

# ---------------------------------------------------
# Patient Details
# ---------------------------------------------------

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    patient_name = st.text_input(

        "Patient Name"

    )

    age_patient = st.number_input(

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
# Heart Parameters
# ---------------------------------------------------

st.subheader("🫀 Heart Disease Parameters")

c1, c2 = st.columns(2)

with c1:

    age = st.number_input(

        "Age",

        1,

        120,

        40

    )

    sex = st.selectbox(

        "Sex",

        [

            0,

            1

        ],

        format_func=lambda x:

        "Female" if x == 0 else "Male"

    )

    cp = st.selectbox(

        "Chest Pain Type",

        [

            0,

            1,

            2,

            3

        ]

    )

    trestbps = st.number_input(

        "Resting Blood Pressure",

        80,

        250,

        120

    )

    chol = st.number_input(

        "Cholesterol",

        100,

        700,

        200

    )

    fbs = st.selectbox(

        "Fasting Blood Sugar >120",

        [

            0,

            1

        ]

    )

    restecg = st.selectbox(

        "Rest ECG",

        [

            0,

            1,

            2

        ]

    )

with c2:

    thalach = st.number_input(

        "Maximum Heart Rate",

        60,

        250,

        150

    )

    exang = st.selectbox(

        "Exercise Induced Angina",

        [

            0,

            1

        ]

    )

    oldpeak = st.number_input(

        "Old Peak",

        0.0,

        10.0,

        1.0

    )

    slope = st.selectbox(

        "Slope",

        [

            0,

            1,

            2

        ]

    )

    ca = st.selectbox(

        "Major Vessels (CA)",

        [

            0,

            1,

            2,

            3,

            4

        ]

    )

    thal = st.selectbox(

        "Thal",

        [

            0,

            1,

            2,

            3

        ]

    )

predict = st.button(

    "❤️ Predict Heart Disease",

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

        age,

        sex,

        cp,

        trestbps,

        chol,

        fbs,

        restecg,

        thalach,

        exang,

        oldpeak,

        slope,

        ca,

        thal

    ]

    with st.spinner("Predicting Heart Disease..."):

        result = predict_heart(input_data)

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

                "Consult a Cardiologist."

            ]

        }

    )

    # ---------------------------------------------------
    # Generate PDF Report
    # ---------------------------------------------------

    pdf_path = report_generator.generate_report(

        patient_name=patient_name,

        age=age_patient,

        gender=gender,

        prediction=prediction,

        confidence=confidence,

        symptoms=[

            f"Age : {age}",

            f"Sex : {'Male' if sex==1 else 'Female'}",

            f"Chest Pain : {cp}",

            f"Resting BP : {trestbps}",

            f"Cholesterol : {chol}",

            f"FBS : {fbs}",

            f"Rest ECG : {restecg}",

            f"Max HR : {thalach}",

            f"Exercise Angina : {exang}",

            f"Old Peak : {oldpeak}",

            f"Slope : {slope}",

            f"CA : {ca}",

            f"Thal : {thal}"

        ],

        description=info["description"],

        precautions=info["precautions"],

        model_used="HeartNet"

    )

    # ---------------------------------------------------
    # Save Prediction
    # ---------------------------------------------------

    db.save_prediction(

        patient_name=patient_name,

        age=age_patient,

        gender=gender,

        disease=prediction,

        confidence=confidence,

        model_used="HeartNet",

        symptoms=[

            str(age),

            str(sex),

            str(cp),

            str(trestbps),

            str(chol),

            str(fbs),

            str(restecg),

            str(thalach),

            str(exang),

            str(oldpeak),

            str(slope),

            str(ca),

            str(thal)

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
    # Heart Parameters Summary
    # ---------------------------------------------------

    st.divider()

    st.subheader("🫀 Heart Parameters Summary")

    summary = {

        "Age": age,

        "Sex": "Male" if sex == 1 else "Female",

        "Chest Pain Type": cp,

        "Resting Blood Pressure": trestbps,

        "Cholesterol": chol,

        "Fasting Blood Sugar >120": "Yes" if fbs == 1 else "No",

        "Rest ECG": restecg,

        "Maximum Heart Rate": thalach,

        "Exercise Induced Angina": "Yes" if exang == 1 else "No",

        "Old Peak": oldpeak,

        "Slope": slope,

        "Major Vessels (CA)": ca,

        "Thal": thal

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

                label="📥 Download Heart Disease Report",

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

    st.success("✅ Heart Disease Analysis Completed")

    st.markdown(f"""
### Patient Summary

**Patient Name:** {patient_name}

**Age:** {age_patient}

**Gender:** {gender}

**Predicted Result:** **{prediction}**

**Confidence:** **{confidence:.2f}%**

**AI Model Used:** HeartNet
""")    