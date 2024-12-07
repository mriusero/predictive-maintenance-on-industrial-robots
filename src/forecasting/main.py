# main.py
import os

import streamlit as st

from .models_base.lstm_based_crack_forecaster.pipeline import lstm_training_pipeline, lstm_testing_pipeline
from .models_base.rul_survival_predictor.pipeline import survival_predictor_training, survival_predictor_prediction


def get_data():
    """
    Function that loads the dataframes from the session state.
    """
    train_df = st.session_state.data.df['train']
    pseudo_test_with_truth_df = st.session_state.data.df['pseudo_test_with_truth']
    test_df = st.session_state.data.df['test']

    return train_df, pseudo_test_with_truth_df, test_df

def handle_phase_one():
    """
    Model management function that runs the pipeline for the rul survival predictor model.
    """
    train_df, pseudo_test_with_truth_df, test_df = get_data()           # Load dataframes
    optimize = st.checkbox('Optimize Hyperparameters', value=False)     # Optimization toggle

    # Run pipeline phase I (TRAINING)
    if st.button('Run training phase I'):
        os.system('clear')
        survival_predictor_training(train_df, pseudo_test_with_truth_df, optimize=optimize)

    # Run pipeline phase I (PREDICTIONS)
    if st.button('Run predictions phase I'):
        os.system('clear')
        survival_predictor_prediction(train_df, test_df)

def handle_phase_two():
    """
    Model management function that runs the pipeline for the lstm crack growth forecast model.
    """
    train_df, pseudo_test_with_truth_df, test_df = get_data()  # Load dataframes
    optimize = st.checkbox('Optimize Hyperparameters', value=False)     # Optimization toggle

    # Run pipeline phase II (TRAINING)
    if st.button('Run training phase II'):
        os.system('clear')
        lstm_training_pipeline(train_df, pseudo_test_with_truth_df, optimize=optimize)

    # Run pipeline phase II (PREDICTIONS)
    if st.button('Run predictions phase II'):
        os.system('clear')
        lstm_testing_pipeline(train_df, test_df, optimize=optimize)