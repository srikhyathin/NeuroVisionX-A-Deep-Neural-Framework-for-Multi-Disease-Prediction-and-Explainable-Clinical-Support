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

from backend.database import NeuroVisionDatabase

# -------------------------------------------------------
# Initialize Database
# -------------------------------------------------------

db = NeuroVisionDatabase()

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------



st.title("📜 Patient Prediction History")

st.markdown("""
View all patient predictions stored in the NeuroVisionX database.
""")

st.divider()

# -----------------------------------------------
# Search
# -----------------------------------------------

search = st.text_input(
    "🔍 Search Patient"
)

refresh = st.button(
    "🔄 Refresh"
)

st.divider()

# -----------------------------------------------
# Load Data
# -----------------------------------------------

if search.strip() == "":

    records = db.get_all_predictions()

else:

    records = db.search_patient(search)

# -----------------------------------------------
# Display Records
# -----------------------------------------------

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
st.divider()

st.subheader("🗑 Delete Record")

record_id = st.number_input(

    "Record ID",

    min_value=1,

    value=1

)

if st.button("Delete Record"):

    db.delete_prediction(record_id)

    st.success("Record deleted successfully.")

st.divider()

st.subheader("⬇ Export CSV")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(

    "Download CSV",

    csv,

    "PatientHistory.csv",

    "text/csv",

    use_container_width=True

)    