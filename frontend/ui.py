import streamlit as st

def apply_theme():

    st.markdown("""
    <style>

    /* Hide Streamlit branding */

    [data-testid="stSidebarHeader"]{
        display:none;
    }

    [data-testid="stSidebarNav"]::before{
        display:none;
    }

    footer{
        visibility:hidden;
    }

    #MainMenu{
        visibility:hidden;
    }

    header{
        visibility:hidden;
    }

    /* Main page */

    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
    }

    /* Buttons */

    .stButton>button{

        width:100%;

        border-radius:12px;

        height:3em;

        border:none;

        background:linear-gradient(90deg,#6A5CFF,#3BA4FF);

        color:white;

        font-weight:bold;

        font-size:16px;

    }

    .stButton>button:hover{

        background:linear-gradient(90deg,#5848ff,#2288ff);

        transform:scale(1.02);

        transition:0.2s;

    }

    /* Metrics */

    [data-testid="metric-container"]{

        border-radius:15px;

        padding:15px;

        border:1px solid rgba(255,255,255,0.08);

        background:rgba(255,255,255,0.03);

    }

    </style>
    """, unsafe_allow_html=True)