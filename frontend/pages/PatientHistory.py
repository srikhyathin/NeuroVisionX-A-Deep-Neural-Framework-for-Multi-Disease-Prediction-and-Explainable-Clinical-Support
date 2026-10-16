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

from sidebar import load_sidebar
from ui import apply_theme

from backend.database import NeuroVisionDatabase

# -------------------------------------------------------
# Initialize Database
# -------------------------------------------------------

db = NeuroVisionDatabase()

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Patient History",
    page_icon="📜",
    layout="wide"
)

apply_theme()


# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("📜 Patient Prediction History")

st.markdown("""
View all patient predictions stored in the NeuroVisionX database.
""")

st.divider()

# -------------------------------------------------------
# Search
# -------------------------------------------------------

search = st.text_input(
    "🔍 Search Patient"
)

refresh = st.button(
    "🔄 Refresh"
)

st.divider()

# -------------------------------------------------------
# Load Records
# -------------------------------------------------------

if search.strip() == "":

    records = db.get_all_predictions()

else:

    records = db.search_patient(search)

# -------------------------------------------------------
# Display Records
# -------------------------------------------------------

df = pd.DataFrame()

if len(records) == 0:

    st.warning("No records found.")

else:

    columns = [

        "ID",
        "Patient",
        "Age",
        "Gender",
        "Disease",
        "Confidence",
        "Model",
        "Symptoms",
        "Report",
        "Time"

    ]

    df = pd.DataFrame(
        records,
        columns=columns
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Total Records : {len(df)}"
    )

# -------------------------------------------------------
# Delete Record
# -------------------------------------------------------

st.divider()

st.subheader("🗑 Delete Record")

record_id = st.number_input(
    "Record ID",
    min_value=1,
    value=1
)

if st.button("🗑 Delete Record"):

    db.delete_prediction(record_id)

    st.success(
        "Record deleted successfully."
    )

    st.rerun()

# -------------------------------------------------------
# Export CSV
# -------------------------------------------------------

st.divider()

st.subheader("⬇ Export CSV")

if df.empty:

    st.info(
        "No records available to export."
    )

else:

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="📥 Download CSV",

        data=csv,

        file_name="PatientHistory.csv",

        mime="text/csv",

        use_container_width=True

    )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.divider()

st.success(
    "✅ Patient history loaded successfully."
)

st.caption(
    "NeuroVisionX • AI-Powered Clinical Decision Support System"
)