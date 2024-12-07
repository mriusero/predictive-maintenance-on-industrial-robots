# main.py
import os
import streamlit as st

from .models_base.rul_survival_predictor.pipeline import survival_predictor_training, survival_predictor_prediction


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

    # Run pipeline phase I (TRAINING)
    if st.button('Run training phase I'):
        os.system('clear')
        survival_predictor_training(train_df, pseudo_test_with_truth_df, optimize=optimize)

    # Run pipeline phase I (PREDICTIONS)
    if st.button('Run predictions phase I'):
        os.system('clear')
        survival_predictor_prediction(train_df, test_df)