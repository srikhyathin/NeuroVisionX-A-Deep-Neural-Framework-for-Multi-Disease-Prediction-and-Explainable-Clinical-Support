import os
import streamlit as st

def load_sidebar():

    logo_path = os.path.join(
        os.path.dirname(__file__),
        "assets",
        "neurovisionx_logo.png"
    )

    st.sidebar.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            padding-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.image(logo_path, width=180)

    st.sidebar.markdown(
        """
        <div style="text-align:center;">
            <h2 style="margin-bottom:0;color:#5B6CFF;">NeuroVisionX</h2>
            <p style="margin-top:0;color:gray;">
                AI-Powered Healthcare
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.divider()