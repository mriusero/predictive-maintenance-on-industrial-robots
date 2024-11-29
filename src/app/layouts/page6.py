import streamlit as st
from src.FailuresDetectModel.main import handle_models

def page_6():
    st.markdown('<div class="header">#6 Prediction_</div>', unsafe_allow_html=True)
    texte = """
    Here is the Prediction phase.
    """
    st.markdown(texte)

    #st.write('### Train DataFrame')
    #st.dataframe(st.session_state.data.get_the('train'))
    #st.write('### Pseudo-test DataFrame')
    #st.dataframe(st.session_state.data.get_the('pseudo_test'))

    handle_models()

