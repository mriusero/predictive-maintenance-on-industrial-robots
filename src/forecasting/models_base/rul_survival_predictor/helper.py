import streamlit as st
import pandas as pd

from src.forecasting.validation.validation import generate_submission_file, calculate_score

from ..configs import SELECTED_VARIABLES


def select_variables(df):
    columns_to_keep = [col for col in df.columns if col not in SELECTED_VARIABLES]
    return df[columns_to_keep]


def analyze(
    model,
    predictions,
    pseudo_test_with_truth_df,
    submission_path,
    model_name,
    step='cross-val'
):
    """
    Helper function to process predictions, compute metrics, save outputs, and generate submission file.

    Parameters:
        model: The predictive model with utility functions.
        predictions: DataFrame containing validation predictions.
        pseudo_test_with_truth_df: DataFrame containing pseudo-test data with true labels.
        submission_path: Path to save outputs.
        model_name: Name of the model for generating submission file.
        step: Current step for processing (default: 'cross-val').

    Returns:
        score: The cross-validation score.
    """
    def add_unique_index(df):
        df['unique_index'] = df.apply(
            lambda row: f"id_{row['item_id']}_&_mth_{row['time (months)']}", axis=1
        )
        df.reset_index(drop=True, inplace=True)
        df.set_index('unique_index', inplace=True)
        return df

    predictions = add_unique_index(predictions)
    pseudo_test_with_truth_df = add_unique_index(pseudo_test_with_truth_df)

    # Merge predictions with ground truth
    predictions_merged = pd.merge(
        predictions,
        pseudo_test_with_truth_df[['label', 'true_rul']],
        on='unique_index',
        how='left'
    )
    predictions_merged.reset_index(drop=True, inplace=True)

    # Compute deducted RUL
    predictions_merged['deducted_rul'] = (
        predictions_merged
        .groupby('item_id', group_keys=False)
        .apply(model.compute_deducted_rul)
        .explode()
        .astype(int)
        .reset_index(drop=True)
    )

    model.save_predictions(model_name, submission_path, step, predictions_merged)   # Save predictions
    generate_submission_file(model_name, submission_path, step)                     # Generate submission file
    score = calculate_score(model_name, submission_path, step)                      # Calculate score
    st.write(f"Validation score for {step}: {score}")

    return predictions_merged