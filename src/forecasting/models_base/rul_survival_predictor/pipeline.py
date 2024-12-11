import datetime
import pandas as pd
import streamlit as st

from src.forecasting.models_base.rul_survival_predictor.configs import MODEL_NAME, SUBMISSION_FOLDER, MODEL_PATH
from src.forecasting.validation.validation import generate_submission_file
from .evaluation import display_results
from .helper import SELECTED_VARIABLES, analyze
from .model import GradientBoostingSurvivalModel
from .optimization import optimize_hyperparameters
from .processing import prepare_train_data, prepare_validation_sets

def survival_predictor_training(train_df: pd.DataFrame, pseudo_test_with_truth_df: pd.DataFrame, optimize: bool):
    """
    Runs the full pipeline for training, optimizing, and evaluating the model with the validation set.
    """
    print("\n" + "=" * 60)
    print(" SURVIVAL PREDICTOR TRAINING")
    print("=" * 60)

    print('1. Processing Data')
    print("-" * 60)
    # Prepare Training Data
    x_train, y_train = prepare_train_data(
        train_df, columns_to_include=SELECTED_VARIABLES
    )
    print(f"training_set: x_train shape {x_train.shape}, y_train shape {y_train.shape}")

    # Prepare Validation Data
    validation_data = prepare_validation_sets(
        n_sets=10,
        reference_df=train_df,
        columns_to_include=SELECTED_VARIABLES
    )
    for key, (x_val, y_val) in validation_data.items():
        print(f"{key}: x_val shape {x_val.shape}, y_val shape {y_val.shape}")

    # Prepare Testing Data
    #x_test, _ = prepare_train_data(
    #    test_df,
    #    columns_to_include=SELECTED_VARIABLES
    #)


    # Initialize Model
    model = GradientBoostingSurvivalModel()

    # Hyperparameter Optimization
    if optimize or (model.best_params is None):
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

    # Save Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_with_timestamp = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pkl"
    try:
        model.save_model(path=model_path_with_timestamp)
        st.success(f"Model saved successfully at {model_path_with_timestamp}")
    except Exception as e:
        st.error(f"Failed to save model: {e}")

def survival_predictor_prediction(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Runs the full pipeline for evaluating the model with the test set.
    """
    print("\n" + "=" * 60)
    print(" SURVIVAL PREDICTOR PREDICTION")
    print("=" * 60)

    # Data Preparation
    x_test, _ = prepare_data(
        test_df,
        reference_df=train_df,
        columns_to_include=SELECTED_VARIABLES
    )

    # Load Model
    model = GradientBoostingSurvivalModel.load_model(path=f'models/rul_survival_predictor/rul_survival_predictor_model.pkl')

    # Prediction on Test Set
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

    # Save Predictions
    model.save_predictions(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step, predictions_df=test_predictions)

    # Generate Submission File
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step)
