import os
import gc
import streamlit as st
from src.core import load_data, DataVisualizer
from .components import github_button


update_message = 'Data loaded'
display = ""

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app_layout():
    from .layouts import page_0, page_1, page_2, page_3, page_4, page_5, page_6

    st.set_page_config(
        page_title="SDA-MACHINE-LEARNING",
        layout='wide',
        initial_sidebar_state="auto",
    )

    load_css()
    st.sidebar.markdown("# --- MACHINE LEARNING ---\n\n"
                        " ## *'Predictive maintenance through failure prediction on robots'*\n")

    page = st.sidebar.radio("Project_", ["#0 Introduction_",
                                         "#1 Exploration_",
                                         "#2 Feature Engineering_",
                                         "#3 Phase I: Survival Prediction_",
                                         "#4 Phase II: Length Forecasting_",
                                         "#5 Fleet Management_",
                                         "#6 Empty_",
                                         ])
    # -- LAYOUT --
    col1, col2 = st.columns([6,4])
    with col1:
        global update_message
        st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)
        st.markdown("#### *'Predictive Maintenance with failures detection on industrial robot'* ")
        colA, colB, colC, colD = st.columns ([1,4,4,3])
        with colA:
            #st.text("")
            github_button('https://github.com/mriusero/predictive-maintenance-on-industrial-robots')
        with colB:
            #st.text("")
            st.text("")
            st.link_button('Kaggle competition : phase I',
                           'https://www.kaggle.com/competitions/predictive-maintenance-for-industrial-robots-i')
        with colC:
            #st.text("")
            st.text("")
            st.link_button('Kaggle competition : phase II',
                           'https://www.kaggle.com/competitions/predictive-maintenance-of-a-robot-ii')
        with colD:
            #st.text("")
            st.text("")
            if st.button('Update data'):
                update_message = load_data()
                st.sidebar.success(f"{update_message}")
                print(update_message)

    with col2:
        st.text("")
        st.text("")
        st.text("")

        data = DataVisualizer()
        st.session_state.data = data

    line_style = """
        <style>
        .full-width-line {
            height: 2px;
            background-color: #FFFFFF; /* Changez la couleur ici (rouge) */
            width: 100%;
            margin: 20px 0;
        }
        </style>
    """
    line_html = '<div class="full-width-line"></div>'

    # Affichage du style et de la ligne
    st.markdown(line_style, unsafe_allow_html=True)
    st.markdown(line_html, unsafe_allow_html=True)

   # st.markdown(f"###### _____________________________________________________________________________________________________________________________________________________")


    if page == "#0 Introduction_":
        page_0()
    elif page == "#1 Exploration_":
        page_1()
    elif page == "#2 Feature Engineering_":
        page_2()
    elif page == "#3 Phase I: Survival Prediction_":
        page_3()
    elif page == "#4 Phase II: Length Forecasting_":
        page_4()
    elif page == "#5 Fleet Management_":
        page_5()
    elif page == "#6 Empty_":
        page_6()

    st.sidebar.markdown("&nbsp;")

    gc.collect()



