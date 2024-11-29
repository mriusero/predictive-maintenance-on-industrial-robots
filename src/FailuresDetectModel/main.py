# main.py
import os
import streamlit as st
from ..FailuresDetectModel import (RandomForestClassifierModel, LSTMModel, GradientBoostingSurvivalModel)

def instance_model(model_name):
    model_classes = {
        'LSTMModel': lambda: LSTMModel(min_sequence_length=2, forecast_months=6),
        'RandomForestClassifierModel': lambda: RandomForestClassifierModel(),
        'GradientBoostingSurvivalModel': lambda: GradientBoostingSurvivalModel()
    }
    if model_name not in model_classes:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model_classes[model_name]()

def filter_columns_exclude(df, columns_to_exclude):
    columns_to_keep = [col for col in df.columns if col not in columns_to_exclude]
    return df[columns_to_keep]

#@st.cache_resource
def ml_pipeline(train_df, pseudo_test_with_truth_df, test_df, optimize):
    """
    Function that executes the three phases of the machine learning pipeline for the selected models.

    Args:
        selected_models (list):                    List of models to run.
        train_df (DataFrame):                      Training dataset.
        pseudo_test_with_truth_df (DataFrame):     Test dataset for cross-validation.
        test_df (DataFrame):                       Final test dataset for predictions.
        optimize (bool):                           Whether to optimize hyperparameters.
    """
    # --------- Gradient Boosting Survival Model ------------
    gb_model = instance_model('GradientBoostingSurvivalModel')
    columns_to_include = ["source", "item_id", "time (months)", "label",                        #"length_measured",
                          "Infant mortality", "Control board failure", "Fatigue crack",         #"Failure mode"
                          "Time to failure (months)", "rul (months)", "failure_month",          #"true_rul",
                          "length_filtered", "beta0", "beta1", "beta2",
                          "rolling_mean_time (months)", "static_mean_time (months)",
                          "rolling_std_time (months)", "static_std_time (months)",
                          "rolling_max_time (months)", "static_max_time (months)",
                          "rolling_min_time (months)", "static_min_time (months)",
                          "rolling_mean_length_filtered", "static_mean_length_filtered",
                          "rolling_std_length_filtered", "static_std_length_filtered",
                          "rolling_max_length_filtered", "static_max_length_filtered",
                          "rolling_min_length_filtered", "static_min_length_filtered",
                          "length_filtered_shift_1",
                          "length_filtered_shift_2", "length_filtered_shift_3",
                          "length_filtered_shift_4", "length_filtered_shift_5",
                          "length_filtered_shift_6", "length_filtered_ratio_1-2",
                          "length_filtered_ratio_2-3", "length_filtered_ratio_3-4",
                          "length_filtered_ratio_4-5", "length_filtered_ratio_5-6",
                          "Trend", "Seasonal", "Residual"]
                          # "rolling_mean_length_measured", "static_mean_length_measured",
                          # "rolling_std_length_measured", "static_std_length_measured",
                          # "rolling_max_length_measured", "static_max_length_measured",
                          # "rolling_min_length_measured", "static_min_length_measured",
                          # "length_measured_shift_1", "length_measured_shift_2",
                          # "length_measured_shift_3", "length_measured_shift_4",
                          # "length_measured_shift_5", "length_measured_shift_6",
                          # "length_measured_ratio_1-2", "length_measured_ratio_2-3",
                          # "length_measured_ratio_3-4", "length_measured_ratio_4-5",
                          # "length_measured_ratio_5-6",

    val_predictions, test_predictions = gb_model.run_full_pipeline(train_df, pseudo_test_with_truth_df, test_df, optimize, columns_to_include)
    col1, col2 = st.columns(2)
    with col1:
        st.write('## Cross val')
        st.dataframe(val_predictions[['item_id',"time (months)", 'length_filtered',
                                      'predicted_failure_now', 'predicted_failure_now_binary',
                                      'predicted_failure_6_months', 'predicted_failure_6_months_binary',
                                      'label_y', 'true_rul',
                                      'deducted_rul']])
    with col2 :
        st.write('## Test final')
        st.dataframe(test_predictions[['item_id', "time (months)", 'length_filtered',
                                    'predicted_failure_now', 'predicted_failure_now_binary',
                                    'predicted_failure_6_months', 'predicted_failure_6_months_binary',
                                    'deducted_rul']])

    st.markdown("# Below are previous test for phase I:\n______________________________________")
    # --------- LSTM Model ------------
    lstm_model = instance_model('LSTMModel')
    lstm_predictions_cross_val, lstm_predictions_final_test = lstm_model.run_full_pipeline(train_df, pseudo_test_with_truth_df, test_df)
#
    st.dataframe(lstm_predictions_cross_val)
    st.dataframe(lstm_predictions_final_test)


def handle_models():
    """
    Model management function that runs the pipeline for each selected model.
    """
    train_df = st.session_state.data.df['train']
    pseudo_test_with_truth_df = st.session_state.data.df['pseudo_test_with_truth']
    test_df = st.session_state.data.df['test']

    optimize = st.checkbox('Optimize Hyperparameters', value=False)

    if st.button('Run predictions phase I'):
        os.system('clear')
        ml_pipeline(train_df, pseudo_test_with_truth_df, test_df, optimize=optimize)

