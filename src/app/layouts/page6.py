import streamlit as st


def page_6():
    st.markdown('<div class="header">#6 Empty_</div>', unsafe_allow_html=True)
    st.markdown("""
    Empty page
    """)

    train_df = st.session_state.data.df['train']
    pseudo_test_with_truth_df = st.session_state.data.df['pseudo_test_with_truth']
    test_df = st.session_state.data.df['test']

    st.write('# Train Data_\n', train_df)
    st.write('# Pseudo Test Data_\n', pseudo_test_with_truth_df)
    st.write('# Test Data_\n', test_df)