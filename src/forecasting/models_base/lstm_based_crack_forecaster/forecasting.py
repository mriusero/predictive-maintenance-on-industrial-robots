import numpy as np
import streamlit as st

from .processing import prepare_test_sequence


def predict_futures_values(model, df):
    if model is None:
        raise ValueError("The model has not been trained.")

    def extract_features(item_data):
        # Print the columns being used for predictions
        print("Extracting features for predictions.")
        print("Columns availables:", item_data.columns.tolist())

        features = {
            'times': item_data['time (months)'].values,
            'length_filtered': item_data['length_filtered'].values,
            'length_measured': item_data['length_measured'].values,
            'rolling_means_filtered': item_data['rolling_mean_length_filtered'].values,
            'rolling_stds_filtered': item_data['rolling_std_length_filtered'].values,
            'rolling_maxs_filtered': item_data['rolling_max_length_filtered'].values,
            'rolling_mins_filtered': item_data['rolling_min_length_filtered'].values,
            #    'rolling_means_measured': item_data['rolling_mean_length_measured'].values,
            #    'rolling_stds_measured': item_data['rolling_std_length_measured'].values,
            #    'rolling_maxs_measured': item_data['rolling_max_length_measured'].values,
            #    'rolling_mins_measured': item_data['rolling_min_length_measured'].values
        }
        return features


    item_indices = df['item_id'].unique()
    all_predictions = []

    with st.spinner('Calculating future values...'):

        progress_bar = st.progress(0)
        num_items = len(item_indices)

        for idx, item_index in enumerate(item_indices):
            item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')

            features = extract_features(item_data)

            print("Columns used:", list(features.keys()))

            last_sequence_padded = prepare_test_sequence(features)
            pred_lengths_filtered, pred_lengths_measured = model.predict(last_sequence_padded)

            print("Prediction shapes: lengths_filtered:", pred_lengths_filtered.shape, "lengths_measured:",
                  pred_lengths_measured.shape)

            combined_predictions = np.column_stack(
                (pred_lengths_filtered.flatten(), pred_lengths_measured.flatten()))
            all_predictions.append(combined_predictions)

            progress_bar.progress((idx + 1) / num_items)

        progress_bar.empty()

    st.success('Future values calculation completed!')
    return all_predictions