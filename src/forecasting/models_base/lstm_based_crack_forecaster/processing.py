import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_train_sequences(df, min_sequence_length=2, forecast_months=6):
    item_indices = df['item_id'].unique()
    sequences = []
    targets_filtered = []
    targets_measured = []

    for item_index in item_indices:
        item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')

        # Print the columns being used for sequence preparation
        print(f"Preparing training sequences for item_id: {item_index}")
        print("Columns used:", item_data.columns.tolist())

        times = item_data['time (months)'].values
        lengths_filtered = item_data['length_filtered'].values
        lengths_measured = item_data['length_measured'].values
        rolling_means_filtered = item_data['rolling_mean_length_filtered'].values
        rolling_stds_filtered = item_data['rolling_std_length_filtered'].values
        rolling_maxs_filtered = item_data['rolling_max_length_filtered'].values
        rolling_mins_filtered = item_data['rolling_min_length_filtered'].values
        # rolling_means_measured = item_data['rolling_mean_length_measured'].values
        # rolling_stds_measured = item_data['rolling_std_length_measured'].values
        # rolling_maxs_measured = item_data['rolling_max_length_measured'].values
        # rolling_mins_measured = item_data['rolling_min_length_measured'].values

        print(f"item_id: {item_index}, Length of data: {len(times)}")

        sequence_length = min_sequence_length

        for i in range(len(times) - sequence_length - forecast_months + 1):
            seq = np.column_stack((
                times[i:i + sequence_length],
                lengths_filtered[i:i + sequence_length],
                lengths_measured[i:i + sequence_length],
                rolling_means_filtered[i:i + sequence_length],
                rolling_stds_filtered[i:i + sequence_length],
                rolling_maxs_filtered[i:i + sequence_length],
                rolling_mins_filtered[i:i + sequence_length],
                #      rolling_means_measured[i:i + sequence_length],
                #      rolling_stds_measured[i:i + sequence_length],
                #      rolling_maxs_measured[i:i + sequence_length],
                #      rolling_mins_measured[i:i + sequence_length]
            ))
            sequences.append(seq)

            target_filtered = lengths_filtered[i + sequence_length:i + sequence_length + forecast_months]
            target_measured = lengths_measured[i + sequence_length:i + sequence_length + forecast_months]

            targets_filtered.append(target_filtered)
            targets_measured.append(target_measured)

    if len(sequences) == 0:
        raise ValueError("No valid sequence was created with the provided data.")

    sequences_padded = np.array(
        pad_sequences(sequences, maxlen=min_sequence_length, padding='post', dtype='float32'))

    targets_filtered = np.array(targets_filtered).reshape(-1, forecast_months)
    targets_measured = np.array(targets_measured).reshape(-1, forecast_months)

    return sequences_padded, {'lengths_filtered_output': targets_filtered,
                              'lengths_measured_output': targets_measured}


def prepare_test_sequence(features, min_sequence_length=2):
    last_sequence = np.column_stack((
        features['times'][-min_sequence_length:],
        features['length_filtered'][-min_sequence_length:],
        features['length_measured'][-min_sequence_length:],
        features['rolling_means_filtered'][-min_sequence_length:],
        features['rolling_stds_filtered'][-min_sequence_length:],
        features['rolling_maxs_filtered'][-min_sequence_length:],
        features['rolling_mins_filtered'][-min_sequence_length:],
        #   features['rolling_means_measured'][-self.min_sequence_length:],
        #   features['rolling_stds_measured'][-self.min_sequence_length:],
        #   features['rolling_maxs_measured'][-self.min_sequence_length:],
        #   features['rolling_mins_measured'][-self.min_sequence_length:]
    ))
    return pad_sequences([last_sequence], maxlen=min_sequence_length, padding='post', dtype='float32')