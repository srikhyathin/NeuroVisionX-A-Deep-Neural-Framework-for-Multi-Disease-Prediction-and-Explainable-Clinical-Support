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

import tempfile
import streamlit as st
from PIL import Image

from backend.predictor import predict_pneumonia
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



st.title("🫁 Pneumonia Detection")

st.markdown("""

Upload a Chest X-ray image.

NeuroVisionX AI will classify whether the patient has Pneumonia.

""")

st.divider()

# ---------------------------------------------------
# Patient Information
# ---------------------------------------------------

st.subheader("👤 Patient Information")

c1, c2 = st.columns(2)

with c1:

    patient_name = st.text_input(
        "Patient Name"
    )

    patient_age = st.number_input(
        "Patient Age",
        1,
        120,
        30
    )

with c2:

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
# Upload Chest X-ray
# ---------------------------------------------------

st.subheader("🩻 Upload Chest X-ray")

uploaded_file = st.file_uploader(

    "Choose Chest X-ray Image",

    type=[

        "jpg",

        "jpeg",

        "png"

    ]

)

predict = st.button(

    "🫁 Detect Pneumonia",

    use_container_width=True

)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if predict:

    if patient_name.strip() == "":

        st.error("Please enter the patient name.")

        st.stop()

    if uploaded_file is None:

        st.error("Please upload a Chest X-ray image.")

        st.stop()

    # ---------------------------------------------------
    # Display Uploaded Image
    # ---------------------------------------------------

    image = Image.open(uploaded_file)

    st.image(

        image,

        caption="Uploaded Chest X-ray",

        use_container_width=True

    )

    # ---------------------------------------------------
    # Save Image Temporarily
    # ---------------------------------------------------

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".jpg"

    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        image_path = tmp_file.name

    # ---------------------------------------------------
    # Prediction
    # ---------------------------------------------------

    with st.spinner("Detecting Pneumonia..."):

        result = predict_pneumonia(image_path)

    if not result["success"]:

        st.error(result["error"])

        os.remove(image_path)

        st.stop()

    prediction = result["prediction"]

    confidence = result["confidence"]

    info = DISEASE_INFO.get(

        prediction,

        {

            "description": "Description unavailable.",

            "precautions": [

                "Consult a Pulmonologist."

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

            "Chest X-ray Uploaded"

        ],

        description=info["description"],

        precautions=info["precautions"],

        model_used="PneumoniaNet"

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

        model_used="PneumoniaNet",

        symptoms=[

            "Chest X-ray"

        ],

        report_path=pdf_path

    )

    # ---------------------------------------------------
    # Display Result
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
    # Chest X-ray Summary
    # ---------------------------------------------------

    st.divider()

    st.subheader("🩻 Chest X-ray Summary")

    st.write(f"**Patient Name:** {patient_name}")
    st.write(f"**Age:** {patient_age}")
    st.write(f"**Gender:** {gender}")

    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.write("**AI Model:** PneumoniaNet (EfficientNet-B0)")

    # ---------------------------------------------------
    # Download PDF Report
    # ---------------------------------------------------

    st.divider()

    st.subheader("📄 Medical Report")

    try:

        with open(pdf_path, "rb") as pdf_file:

            st.download_button(

                label="📥 Download Pneumonia Report",

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

    st.success("✅ Pneumonia Analysis Completed Successfully")

    st.markdown(f"""

### Patient Summary

**Patient Name:** {patient_name}

**Age:** {patient_age}

**Gender:** {gender}

**Predicted Result:** **{prediction}**

**Confidence:** **{confidence:.2f}%**

**AI Model Used:** PneumoniaNet (EfficientNet-B0)

""")

    # ---------------------------------------------------
    # Delete Temporary Image
    # ---------------------------------------------------

    try:

        os.remove(image_path)

    except Exception:

        pass   