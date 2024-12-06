import pandas as pd
import streamlit as st

from .model import GradientBoostingSurvivalModel
from .processing import prepare_data
from .optimization import optimize_hyperparameters
from .helper import SELECTED_VARIABLES, analyze
from .evaluation import display_results
from ..configs import MODEL_NAME, SUBMISSION_FOLDER

from src.forecasting.validation.validation import generate_submission_file, calculate_score

def survival_predictor_pipeline(train_df: pd.DataFrame, pseudo_test_with_truth_df: pd.DataFrame, test_df: pd.DataFrame, optimize: bool):
    """
    Runs the full pipeline for training, optimizing, and evaluating the model.
    """
    print('Running GradientBoostingSurvivalModel Pipeline...')

    # Prepare Data
    print('...data processing...')
    x_train, y_train = prepare_data(
        train_df, columns_to_include=SELECTED_VARIABLES
    )
    x_val, y_val = prepare_data(
        pseudo_test_with_truth_df.drop(columns=['true_rul']),
        reference_df=train_df,
        columns_to_include=SELECTED_VARIABLES
    )
    x_test, _ = prepare_data(
        test_df,
        reference_df=train_df,
        columns_to_include=SELECTED_VARIABLES
    )

    print('...model initialization...')         # Initialize Model
    model = GradientBoostingSurvivalModel()

    # Hyperparameters
    if optimize or (model.best_params is None):
        print('...optimization asked or best_params is None, hyperparameter optimization...')
        model.best_params = optimize_hyperparameters(x_train, y_train)
        st.success("Hyperparameters have been optimized and saved.")

    # Train Model
    model.train(x_train, y_train)

    # Validation
    val_predictions = model.predict(x_val, columns_to_include=SELECTED_VARIABLES)
    val_predictions_merged = analyze(
        model=model,
        predictions=val_predictions,
        pseudo_test_with_truth_df=pseudo_test_with_truth_df,
        submission_path=SUBMISSION_FOLDER,
        model_name=MODEL_NAME,
        step='cross-val'
    )
    display_results(x_train, val_predictions_merged.sort_values(['item_id', 'time (months)'], ascending=True))

    # Test
    step = 'final-test'
    test_predictions = model.predict(x_test, columns_to_include=SELECTED_VARIABLES)
    test_predictions['deducted_rul'] = (
        test_predictions
        .groupby('item_id', group_keys=False)
        .apply(model.compute_deducted_rul)
        .explode()
        .astype(int)
        .reset_index(drop=True)
    )
    model.save_predictions(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step, predictions_df=test_predictions)
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step)
    final_score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step)
    st.write(f"Final score: {final_score}")

    return val_predictions_merged, test_predictions