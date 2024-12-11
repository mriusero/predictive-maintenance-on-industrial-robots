import streamlit as st
from src.forecasting.main import handle_phase_two


def page_4():
    st.markdown('<div class="header">#4 Phase II: Crack Growth Forecasting_</div>', unsafe_allow_html=True)
    st.markdown("""
    Here is the page dedicated to the second phase of the project.
    """)

    st.image(f'models/lstm_based_crack_forecaster/model_visualization.png', caption="Model architecture", use_column_width=True)

    handle_phase_two()