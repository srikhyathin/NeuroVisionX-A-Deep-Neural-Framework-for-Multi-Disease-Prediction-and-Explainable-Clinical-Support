from pathlib import Path
import streamlit as st

# --------------------------------------------------------
# Page Config
# --------------------------------------------------------

st.set_page_config(
    page_title="NeuroVisionX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
PAGES = BASE_DIR / "pages"
ASSETS = BASE_DIR / "assets"

# --------------------------------------------------------
# Sidebar Branding
# --------------------------------------------------------

with st.sidebar:

    st.image(
        str(ASSETS / "neurovisionx_logo.png"),
        width=180
    )

    st.markdown(
        """
        <div style="text-align:center">

        <h2 style="margin-bottom:0;color:#5B6CFF;">
        NeuroVisionX
        </h2>

        <p style="margin-top:0;color:gray;">
        AI-Powered Healthcare Platform
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

# --------------------------------------------------------
# Navigation
# --------------------------------------------------------

navigation = st.navigation({

    "Disease Prediction":[

        st.Page(PAGES/"Home.py",
                title="Home",
                icon="🏠"),

        st.Page(PAGES/"Symptoms.py",
                title="Symptoms",
                icon="🩺"),

        st.Page(PAGES/"Heart.py",
                title="Heart",
                icon="❤️"),

        st.Page(PAGES/"Diabetes.py",
                title="Diabetes",
                icon="🩸"),

        st.Page(PAGES/"BrainTumor.py",
                title="Brain Tumor",
                icon="🧠"),

        st.Page(PAGES/"Pneumonia.py",
                title="Pneumonia",
                icon="🫁")

    ],

    "Management":[

        st.Page(PAGES/"PatientHistory.py",
                title="Patient History",
                icon="📜"),

        st.Page(PAGES/"Analytics.py",
                title="Analytics",
                icon="📊")

    ],

    "Utilities":[

        st.Page(PAGES/"Explainability.py",
                title="Explainability",
                icon="🔍"),

        st.Page(PAGES/"About.py",
                title="About",
                icon="ℹ️")

    ]

})

navigation.run()