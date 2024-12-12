# main.py
import os

import streamlit as st

from .models_base.fleet_management_predictor.pipeline import fleet_management_pipeline
from .models_base.lstm_based_crack_forecaster.pipeline import lstm_training_pipeline, lstm_validation_pipeline, \
    lstm_testing_pipeline
from .models_base.rul_survival_predictor.pipeline import survival_predictor_pipeline


def handle_phase_one():
    """
    Model management function that runs the pipeline for the rul survival predictor model.
    """
    optimize = st.checkbox('Optimize Hyperparameters', value=False)     # Optimization toggle

    # Run pipeline phase I
    if st.button('Run pipeline phase I'):
        os.system('clear')
        survival_predictor_pipeline(optimize=optimize)


#def handle_phase_two():
#    """
#    Model management function that runs the pipeline for the lstm crack growth forecast model.
#    """
#    optimize = st.checkbox('Optimize Hyperparameters', value=False)     # Optimization toggle
#
#    # Run pipeline phase II (TRAINING)
#    if st.button('Run training phase II'):
#        os.system('clear')
#        lstm_training_pipeline(train_df, optimize=optimize)
#
#    # Run pipeline phase II (VALIDATION)
#    if st.button('Run validation phase II'):
#        os.system('clear')
#        lstm_validation_pipeline(train_df, pseudo_test_with_truth_df, optimize=optimize)
#
#    # Run pipeline phase II (PREDICTIONS)
#    if st.button('Run predictions phase II'):
#        os.system('clear')
#        lstm_testing_pipeline(train_df, test_df, optimize=optimize)

#def handle_fleet_management():
#    """
#    Model management function that runs the pipeline for the fleet management.
#    """
#    train_df, pseudo_test_with_truth_df, test_df = get_data()  # Load dataframes
#    fleet_management_pipeline(train_df, pseudo_test_with_truth_df, test_df)
