# main.py
import os
import streamlit as st

from .models_base.rul_survival_predictor import survival_predictor_pipeline


def handle_models():
    """
    Model management function that runs the pipeline for each selected model.
    """
    # Load dataframes
    train_df = st.session_state.data.df['train']
    pseudo_test_with_truth_df = st.session_state.data.df['pseudo_test_with_truth']
    test_df = st.session_state.data.df['test']

    # Optimization toggle
    optimize = st.checkbox('Optimize Hyperparameters', value=False)

    # Run pipeline phase I
    if st.button('Run predictions phase I'):
        os.system('clear')

        survival_predictor_pipeline(train_df, pseudo_test_with_truth_df, test_df, optimize=optimize)
