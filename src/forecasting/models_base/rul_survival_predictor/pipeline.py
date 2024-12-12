import datetime
import streamlit as st
import numpy as np
import tqdm as tqdm
import pandas as pd

from sklearn.metrics import classification_report

from .configs import MODEL_NAME, SUBMISSION_FOLDER, MODEL_PATH, SELECTED_VARIABLES, N_VAL_SETS
from .processing import prepare_data
from .evaluation import measure_performance_and_plot, calculate_combined_metrics
from .helper import analyze
from .model import GradientBoostingSurvivalModel
from .optimization import optimize_hyperparameters

from src.forecasting.validation.validation import generate_submission_file

def survival_predictor_pipeline(optimize: bool):
    """
    Runs the full pipeline for training, optimizing, and evaluating the model with the validation set.
    """
    print("\n" + "=" * 60)
    print(" SURVIVAL PREDICTOR TRAINING")
    print("=" * 60)

    print('1. Data Processing')
    print("-" * 60)
    data = prepare_data(
        selected_variables=SELECTED_VARIABLES,
        n_validation_sets=N_VAL_SETS
    )

    print('\n2. Model Training')
    print("-" * 60)
    model = GradientBoostingSurvivalModel()
    if optimize or (model.best_params is None):
        model.best_params = optimize_hyperparameters(data['x_train'],data['y_train'])
        st.success("Hyperparameters have been optimized and saved.")
    model.train(data['x_train'], data['y_train'])


    print('\n3. Cross-Validation')
    print("-" * 60)
    cross_val_predictions = []
    for i in range (N_VAL_SETS):
        x_val, y_val = data['validation_data'][f'validation_set_{i + 1}']
        val_predictions = model.predict(x_val, columns_to_include=SELECTED_VARIABLES)
        cross_val_predictions.append(val_predictions)


    print('\n4. Evaluation')
    print("-" * 60)
    all_y_true = []
    all_y_pred = []
    for i in range(N_VAL_SETS):
        val_predictions = cross_val_predictions[i]
        val_predictions_merged = analyze(
            model=model,
            predictions=val_predictions,
            pseudo_test_with_truth_df=data['truth_data'][f'validation_set_{i + 1}'],
            submission_path=SUBMISSION_FOLDER,
            model_name=MODEL_NAME,
            step=f'cross-val_{i+1}'
        )
        y_true = val_predictions_merged['label_y'].values
        y_pred = val_predictions_merged['predicted_failure_6_months_binary'].values

        report = classification_report(y_true, y_pred, output_dict=True)
        print(f"-- Classification Report for Validation Set {i + 1} --\n" + '_' * 45)
        print(pd.DataFrame(report).transpose())
        print("\n")

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    measure_performance_and_plot(all_y_true, all_y_pred)


    print('\n5. Model Saving')
    print("-" * 60)
    # Save Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_with_timestamp = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pkl"
    try:
        model.save_model(path=model_path_with_timestamp)
        print(f"Model saved successfully at {model_path_with_timestamp}")
        st.toast(f"Model saved successfully at {model_path_with_timestamp}")
    except Exception as e:
        st.error(f"Failed to save model: {e}")


    print('\n6. Final test set prediction')
    print("-" * 60)
    model = GradientBoostingSurvivalModel.load_model(path=f'models/rul_survival_predictor/rul_survival_predictor_model.pkl')
    step = 'phase-1'
    test_predictions = model.predict(data['x_test'], columns_to_include=SELECTED_VARIABLES)

    test_predictions['deducted_rul'] = (
        test_predictions
        .groupby('item_id', group_keys=False)
        .apply(model.compute_deducted_rul)
        .explode()
        .astype(int)
        .reset_index(drop=True)
    )

    print('\n7. Submission')
    print("-" * 60)
    model.save_predictions(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step, predictions_df=test_predictions)
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step=step)
