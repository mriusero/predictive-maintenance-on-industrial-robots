import json
import numpy as np
import streamlit as st

from .configs import MIN_SEQUENCE_LENGTH, FEATURE_COLUMNS
from .processing import prepare_sequences


def predict_future_values(model, df, output_file_path=None):
    """
    Predict future values for each item in the dataset and optionally save to JSON.
    Parameters:
        model: Trained Keras model used for predictions.
        df: pd.DataFrame containing the dataset.
        output_file_path: (Optional) Path to save predictions as a JSON file.
    Returns:
        all_predictions: List of dictionaries with predictions for each item.
    """
    if model is None:
        raise ValueError("The model has not been trained.")

    item_indices = df['item_id'].unique()
    all_predictions = []

    with st.spinner('Calculating future values...'):
        for item_index in item_indices:
            item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')  # Filter and sort data for the current item

            last_sequence_padded = prepare_sequences(           # Prepare the last sequence for prediction
                data=item_data,
                mode='test',
                min_sequence_length=MIN_SEQUENCE_LENGTH,
                feature_columns=FEATURE_COLUMNS
            )

            if last_sequence_padded.shape[1] != MIN_SEQUENCE_LENGTH or last_sequence_padded.ndim != 3:
                raise ValueError(
                    f"Prepared sequence for item {item_index} has incorrect shape: {last_sequence_padded.shape}. "
                    f"Expected shape: (1, {MIN_SEQUENCE_LENGTH}, len(FEATURE_COLUMNS)).")

            if np.any(np.isnan(last_sequence_padded)):
                raise ValueError(f"Prepared sequence for item {item_index} contains NaN values.")

            batch_predictions = model.predict(last_sequence_padded)  # batch_predictions will be a dictionary

            if isinstance(batch_predictions, dict):
                combined_predictions = {
                    'item_id': int(item_index),
                    'length_filtered': batch_predictions['length_filtered'].flatten().tolist(),
                    'length_measured': batch_predictions['length_measured'].flatten().tolist(),
                    'Infant mortality': batch_predictions['Infant_mortality'].flatten().tolist(),
                    'Control board failure': batch_predictions['Control_board_failure'].flatten().tolist(),
                    'Fatigue crack': batch_predictions['Fatigue_crack'].flatten().tolist(),
                }
            else:
                raise ValueError(f"Unexpected structure of batch_predictions: {type(batch_predictions)}")

            all_predictions.append(combined_predictions)

    st.success('Future values calculation completed!')

    if output_file_path:
        with open(output_file_path, 'w') as json_file:
            json.dump(all_predictions, json_file, indent=4)
        print(f"Predictions saved to {output_file_path}")


