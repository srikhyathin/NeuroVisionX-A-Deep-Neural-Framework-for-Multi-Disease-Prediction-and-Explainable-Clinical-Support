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
import matplotlib.pyplot as plt

from backend.database import NeuroVisionDatabase

# -------------------------------------------------------
# Initialize Database
# -------------------------------------------------------

db = NeuroVisionDatabase()

# -------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------



st.title("📊 NeuroVisionX Analytics Dashboard")

st.markdown("""
Visualize prediction statistics and AI model performance.
""")

st.divider()

records = db.get_all_predictions()

if len(records) == 0:

    st.warning("No prediction history available.")

    st.stop()

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

df = pd.DataFrame(records, columns=columns)
st.subheader("📈 Overall Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Total Predictions",

        len(df)

    )

with col2:

    st.metric(

        "Average Confidence",

        f"{df['Confidence'].mean():.2f}%"

    )

with col3:

    st.metric(

        "Highest Confidence",

        f"{df['Confidence'].max():.2f}%"

    )

with col4:

    st.metric(

        "Unique Diseases",

        df["Disease"].nunique()

    )

st.divider()
st.subheader("📊 Disease Distribution")

disease_counts = df["Disease"].value_counts()

fig, ax = plt.subplots(figsize=(7,5))

ax.bar(

    disease_counts.index,

    disease_counts.values

)

ax.set_xlabel("Disease")

ax.set_ylabel("Count")

ax.set_title("Disease Distribution")

plt.xticks(rotation=45)

st.pyplot(fig)
st.divider()

st.subheader("🥧 Disease Percentage")

fig2, ax2 = plt.subplots(figsize=(7,7))

ax2.pie(

    disease_counts.values,

    labels=disease_counts.index,

    autopct="%1.1f%%"

)

ax2.set_title("Prediction Percentage")

st.pyplot(fig2)

st.divider()

st.subheader("🤖 AI Model Usage")

model_counts = df["Model"].value_counts()

fig3, ax3 = plt.subplots(figsize=(7,5))

ax3.bar(

    model_counts.index,

    model_counts.values

)

ax3.set_title("Model Usage")

ax3.set_ylabel("Predictions")

plt.xticks(rotation=20)

st.pyplot(fig3)

st.divider()

st.subheader("📋 Recent Predictions")

st.dataframe(

    df.head(20),

    use_container_width=True,

    hide_index=True

)