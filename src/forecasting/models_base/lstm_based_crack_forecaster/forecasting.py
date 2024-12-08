import numpy as np
import streamlit as st

from .processing import prepare_test_sequence


def predict_futures_values(model, df, feature_columns, target_columns):
    """
    Predict future values for each item in the dataset.
    Parameters:
        model: Trained Keras model used for predictions.
        df: pd.DataFrame containing the dataset.
        feature_columns: List of columns to include as features.
        target_columns: List of target column names.
    Returns:
        all_predictions: List of arrays with predictions for all items.
    """
    if model is None:
        raise ValueError("The model has not been trained.")

    def extract_features(item_data):
        """
        Extracts the features required for predictions based on the provided feature columns.
        Parameters:
            item_data: Subset of the dataframe for a specific item.
        Returns:
            features: Dictionary containing the features for the test sequence.
        """
        return {col: item_data[col].values for col in feature_columns}

    item_indices = df['item_id'].unique()
    all_predictions = []

    with st.spinner('Calculating future values...'):

        progress_bar = st.progress(0)
        num_items = len(item_indices)

        for idx, item_index in enumerate(item_indices):
            item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')

            features = extract_features(item_data)
            last_sequence_padded = prepare_test_sequence(features, feature_columns=feature_columns)
            predictions = model.predict(last_sequence_padded)

            combined_predictions = {}
            for target, pred in zip(target_columns, predictions):
                combined_predictions[target] = pred.flatten()

            all_predictions.append(combined_predictions)

            progress_bar.progress((idx + 1) / num_items)

        progress_bar.empty()

    st.success('Future values calculation completed!')
    return all_predictions