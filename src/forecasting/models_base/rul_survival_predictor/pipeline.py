import sys
import pandas as pd
import streamlit as st
import datetime

from .model import GradientBoostingSurvivalModel
from .processing import prepare_data
from .optimization import optimize_hyperparameters
from .helper import SELECTED_VARIABLES, analyze
from .evaluation import display_results
from src.forecasting.models_base.rul_survival_predictor.configs import MODEL_NAME, SUBMISSION_FOLDER, MODEL_PATH

from src.forecasting.validation.validation import generate_submission_file, calculate_score

def log_step(step_message, completed=False):
    """Logs a single step with an optional completion message, replacing the current line."""

    sys.stdout.write("\033[K") # Clear the current line
    symbol = "✅ " if completed else "... "
    print(f"{symbol} {step_message}", end="\r" if not completed else "\n")
    sys.stdout.flush()


def survival_predictor_training(train_df: pd.DataFrame, pseudo_test_with_truth_df: pd.DataFrame, optimize: bool):
    """
    Runs the full pipeline for training, optimizing, and evaluating the model with the validation set.
    """
    print("\n" + "=" * 60)
    print(" SURVIVAL PREDICTOR TRAINING")
    print("=" * 60)

    # Prepare Data
    log_step("Preparing training data...")
    x_train, y_train = prepare_data(
        train_df, columns_to_include=SELECTED_VARIABLES
    )
    log_step(f"Training data prepared: {len(x_train)} samples.", completed=True)

    log_step("Preparing validation data...")
    x_val, y_val = prepare_data(
        pseudo_test_with_truth_df.drop(columns=['true_rul']),
        reference_df=train_df,
        columns_to_include=SELECTED_VARIABLES
    )
    log_step(f"Validation data prepared: {len(x_val)} samples.", completed=True)

    # Initialize Model
    log_step("Initializing the Gradient Boosting Survival Model...")
    model = GradientBoostingSurvivalModel()
    log_step("Model initialized.", completed=True)

    # Hyperparameter Optimization
    if optimize or (model.best_params is None):
        log_step("Optimizing hyperparameters...")
        model.best_params = optimize_hyperparameters(x_train, y_train)
        log_step(f"Hyperparameters optimized: {model.best_params}", completed=True)
        st.success("Hyperparameters have been optimized and saved.")

    # Train Model
    log_step("Training the model...")
    model.train(x_train, y_train)
    log_step("Model training completed.", completed=True)

    # Validation
    log_step("Running predictions on the validation set...")
    val_predictions = model.predict(x_val, columns_to_include=SELECTED_VARIABLES)
    log_step("Validation predictions completed.", completed=True)

    log_step("Analyzing validation results...")
    val_predictions_merged = analyze(
        model=model,
        predictions=val_predictions,
        pseudo_test_with_truth_df=pseudo_test_with_truth_df,
        submission_path=SUBMISSION_FOLDER,
        model_name=MODEL_NAME,
        step='cross-val'
    )
    log_step("Validation results analyzed.\n", completed=True)

    display_results(x_train, val_predictions_merged.sort_values(['item_id', 'time (months)'], ascending=True))

    # Save Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_with_timestamp = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pkl"
    try:
        model.save_model(path=model_path_with_timestamp)
        log_step(f"Model saved successfully at {model_path_with_timestamp}", completed=True)
        st.success(f"Model saved successfully at {model_path_with_timestamp}")
    except Exception as e:
        log_step(f"Failed to save model: {e}", completed=True)
        st.error(f"Failed to save model: {e}")


def survival_predictor_prediction(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Runs the full pipeline for evaluating the model with the test set.
    """
    print("\n" + "=" * 60)
    print(" SURVIVAL PREDICTOR PREDICTION")
    print("=" * 60)

    # Data Preparation
    log_step("Preparing test data...")
    x_test, _ = prepare_data(
        test_df,
        reference_df=train_df,
        columns_to_include=SELECTED_VARIABLES
    )
    log_step(f"Test data prepared: {len(x_test)} samples.", completed=True)

    # Load Model
    model = GradientBoostingSurvivalModel.load_model(path=f'models/rul_survival_predictor/rul_survival_predictor_model.pkl')
    log_step(f"Model {MODEL_NAME} logged successfully.", completed=True)

    # Prediction on Test Set
    step = 'final-test'
    log_step("Running predictions on the test set...")
    test_predictions = model.predict(x_test, columns_to_include=SELECTED_VARIABLES)
    log_step("Test predictions completed.", completed=True)

    log_step("Calculating deducted RUL...")
    test_predictions['deducted_rul'] = (
        test_predictions
        .groupby('item_id', group_keys=False)
        .apply(model.compute_deducted_rul)
        .explode()
        .astype(int)
        .reset_index(drop=True)
    )
    log_step("Deducted RUL calculations completed.", completed=True)

    # Save Predictions
    log_step("Saving predictions...")
    model.save_predictions(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step, predictions_df=test_predictions)
    log_step("Predictions saved successfully.", completed=True)

    # Generate Submission File
    log_step("Generating submission file...")
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step)
    log_step("Submission file generated successfully.", completed=True)

    # Calculate Final Score
    log_step("Calculating final score...")
    final_score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step)
    log_step(f"Final score calculated: {final_score}", completed=True)
    st.write(f"Final score: `{final_score}`")
