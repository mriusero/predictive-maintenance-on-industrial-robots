import streamlit as st
from src.forecasting.main import handle_phase_two


def page_4():
    st.markdown('<div class="header">#4 Phase II_</div>', unsafe_allow_html=True)
    st.markdown("""
    Here is the page dedicated to the second phase of the project.
    """)

    handle_phase_two()