import os
import streamlit as st

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app_layout():
    from .content import page_0, page_1, page_2, page_3, page_4

    st.set_page_config(
        page_title="SDA-MACHINE-LEARNING",
        page_icon="",
        layout='wide',
        initial_sidebar_state="auto",
        menu_items={
            'About': "#Github Repository :\n\nhttps://github.com/mriusero/projet-sda-machine-learning/blob/main/README.md"
        }
    )

    load_css()
    page = st.sidebar.radio("Overview", ["#0 Introduction_",
                                         "#1 Exploration & Cleaning_",
                                         "#2 Feature Engineering_",
                                         "#3 Training_",
                                         "#4 Prediction_",
                                         ])

    if page == "#0 Introduction_":
        page_0()
    elif page == "#1 Exploration & Cleaning_":
        page_1()
    elif page == "#2 Feature Engineering_":
        page_2()
    elif page == "#3 Training_":
        page_3()
    elif page == "#4 Prediction_":
        page_4()









