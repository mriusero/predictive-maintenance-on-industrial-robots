# processing.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .configs import MIN_SEQUENCE_LENGTH, FORECAST_MONTHS, FEATURE_COLUMNS, TARGET_COLUMNS


def extract_features(data, feature_columns, start_index, end_index):
    """
    Extracts the specified features for a given range from the data.
    Parameters:
        data: pd.DataFrame or dict, the data to extract features from.
        feature_columns: list of str, the columns to extract.
        start_index: int, start index for slicing.
        end_index: int, end index for slicing.
    Returns:
        A numpy array of extracted features.
    """
    if isinstance(data, dict):  # For test mode with dictionary input
        return np.column_stack([
            data[col][start_index:end_index] for col in feature_columns
        ])
    else:  # For train mode with DataFrame input
        return np.column_stack([
            data[col].values[start_index:end_index] for col in feature_columns
        ])


def prepare_sequences(data, mode, min_sequence_length=MIN_SEQUENCE_LENGTH, forecast_months=FORECAST_MONTHS,
                      feature_columns=FEATURE_COLUMNS, target_columns=TARGET_COLUMNS):
    """
    Prepares sequences for training or testing.
    Parameters:
        data: pd.DataFrame or dict, the input data.
        mode: str, either 'train' or 'test'.
        min_sequence_length: int, minimum length of input sequences.
        forecast_months: int, number of months to forecast (only for 'train' mode).
        feature_columns: list of str, columns to include in the feature sequences.
        target_columns: list of str, columns to use as targets (only for 'train' mode).
    Returns:
        For 'train' mode: Tuple (sequences_padded, targets_dict)
        For 'test' mode: Numpy array of the test sequence.
    """
    if feature_columns is None:
        raise ValueError("feature_columns must be provided.")

    if mode == 'train':
        if target_columns is None:
            raise ValueError("target_columns must be provided in 'train' mode.")

        item_indices = data['item_id'].unique()
        sequences = []
        targets = {col: [] for col in target_columns}

        for item_index in item_indices:

            item_data = data[data['item_id'] == item_index].sort_values(by='time (months)')     # Select and sort data for the current item
            num_rows = len(item_data)

            for i in range(num_rows - min_sequence_length - forecast_months + 1):

                seq = extract_features(item_data, feature_columns, i, i + min_sequence_length)      # Extract sequence features
                sequences.append(seq)

                for col in target_columns:      # Extract targets for forecast
                    target = item_data[col].values[i + min_sequence_length:i + min_sequence_length + forecast_months]
                    targets[col].append(target)

        if not sequences:
            raise ValueError("No valid sequence was created with the provided data.")

        sequences_padded = np.array(
            pad_sequences(sequences, maxlen=min_sequence_length, padding='post', dtype='float32')
        )
        targets_dict = {
            col: np.array(values).reshape(-1, forecast_months) for col, values in targets.items()
        }
        print(f"Shape of X: {sequences_padded.shape}")  # (nb_sequences, min_sequence_length, nb_features)
        for key, value in targets_dict.items():
            print(f"Shape of Y[{key}]: {value.shape}")  # (nb_sequences, forecast_months)

        return sequences_padded, targets_dict


    elif mode == 'test':        # Extract the sequence for testing

        sequence = extract_features(data, feature_columns, -min_sequence_length, None)
        return pad_sequences(
            [sequence], maxlen=min_sequence_length, padding='post', dtype='float32'
        )
    else:
        raise ValueError("Invalid mode. Choose either 'train' or 'test'.")