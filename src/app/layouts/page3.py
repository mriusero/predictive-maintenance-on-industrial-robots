import streamlit as st

from src.forecasting.main import handle_models


def page_3():

    st.markdown('<div class="header">#3 Phase I_</div>', unsafe_allow_html=True)
    st.markdown("""
    Here is the page dedicated to the first phase of the project.
    """)

    handle_models()