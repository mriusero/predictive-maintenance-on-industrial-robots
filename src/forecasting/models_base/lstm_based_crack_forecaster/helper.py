import pandas as pd
import numpy as np

from src.forecasting.preprocessing.features import FeatureAdder


def add_predictions_to_data(df, predictions, min_sequence_length=2):
    def prepare_initial_data(item_data, item_index, source):
        times = item_data['time (months)'].values
        lengths_filtered = item_data['length_filtered'].values
        lengths_measured = item_data['length_measured'].values

        features = {
            'rolling_means_filtered': item_data['rolling_mean_length_filtered'].values,
            'rolling_stds_filtered': item_data['rolling_std_length_filtered'].values,
            'rolling_maxs_filtered': item_data['rolling_max_length_filtered'].values,
            'rolling_mins_filtered': item_data['rolling_min_length_filtered'].values,
            #  'rolling_means_measured': item_data['rolling_mean_length_measured'].values,
            #  'rolling_stds_measured': item_data['rolling_std_length_measured'].values,
            #  'rolling_maxs_measured': item_data['rolling_max_length_measured'].values,
            #  'rolling_mins_measured': item_data['rolling_min_length_measured'].values
        }

        data_dict = {
            'item_id': item_index,
            'time (months)': times,
            'length_filtered': lengths_filtered,
            'length_measured': lengths_measured,
            'source': source
        }
        data_dict.update(features)

        # if scenario[:] == 'Scenario2':
        #    data_dict.update({
        #        'label': item_data['label'].values,
        #        'true_rul': item_data['true_rul'].values
        #    })
        return pd.DataFrame(data_dict)

    item_indices = df['item_id'].unique()
    extended_data = []

    for idx, item_index in enumerate(item_indices):
        item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')
        max_time = np.max(item_data['time (months)'].values)
        forecast_length = len(predictions[idx])
        future_times = np.arange(np.ceil(max_time + 1), np.ceil(max_time + 1) + forecast_length)

        future_lengths_filtered = predictions[idx][:, 0]
        future_lengths_measured = predictions[idx][:, 1]

        initial_data = prepare_initial_data(item_data, item_index, source=0)
        forecast_data = pd.DataFrame({
            'item_id': item_index,
            'time (months)': future_times,
            'length_filtered': future_lengths_filtered,
            'length_measured': future_lengths_measured,
            'source': 1
        })
        extended_data.append(pd.concat([initial_data, forecast_data]))

    if not extended_data:
        raise ValueError("No extended data was created with the provided predictions.")

    df_extended = pd.concat(extended_data).reset_index(drop=True)
    df_extended['crack_failure'] = (
            (df_extended['length_measured'] >= 0.85) | (df_extended['length_filtered'] >= 0.85)).astype(int)

    feature_adder = FeatureAdder(min_sequence_length=min_sequence_length)
    df_extended = feature_adder.add_features(df_extended, particles_filtery=False)

    # df_extended['item_id'] = df_extended['item_index'].astype(str)
    # df_extended.loc[:, 'item_index'] = df_extended['item_index'].apply(lambda x: f'item_{x}')

    return df_extended