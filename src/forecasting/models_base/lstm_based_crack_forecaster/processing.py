import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def extract_features(item_data, feature_columns, start_index, end_index):
    """
    Extracts the specified features for a given range from the item data.
    Parameters:
        item_data: pd.DataFrame, the data for a specific item.
        feature_columns: list of str, the columns to extract.
        start_index: int, start index for slicing.
        end_index: int, end index for slicing.
    Returns:
        A numpy array of extracted features.
    """
    return np.column_stack([item_data[col].values[start_index:end_index] for col in feature_columns])


def prepare_train_sequences(df, min_sequence_length=2, forecast_months=6, feature_columns=None, target_columns=None):
    """
    Prepares training sequences and targets from the given dataframe.
    Parameters:
        df: pd.DataFrame, the input data.
        min_sequence_length: int, minimum length of input sequences.
        forecast_months: int, number of months to forecast.
        feature_columns: list of str, columns to include in the feature sequences.
        target_columns: list of str, columns to use as targets.
    Returns:
        Tuple (sequences_padded, targets_dict)
    """
    if feature_columns is None or target_columns is None:
        raise ValueError("Both feature_columns and target_columns must be provided.")

    item_indices = df['item_id'].unique()
    sequences = []
    targets = {col: [] for col in target_columns}

    for item_index in item_indices:

        item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')     # Select and sort data for the current item
        num_rows = len(item_data)

        for i in range(num_rows - min_sequence_length - forecast_months + 1):           # Generate sequences

            seq = extract_features(item_data, feature_columns, i, i + min_sequence_length)      # Extract sequence features
            sequences.append(seq)

            for col in target_columns:                                                          # Extract targets for forecast
                target = item_data[col].values[i + min_sequence_length:i + min_sequence_length + forecast_months]
                targets[col].append(target)

    if not sequences:
        raise ValueError("No valid sequence was created with the provided data.")

    sequences_padded = np.array(        # Pad sequences for consistent dimensions
        pad_sequences(sequences, maxlen=min_sequence_length, padding='post', dtype='float32')
    )
    targets_dict = {col: np.array(values).reshape(-1, forecast_months) for col, values in targets.items()}      # Convert targets to numpy arrays

    return sequences_padded, targets_dict


def prepare_test_sequence(features, min_sequence_length=2, feature_columns=None):
    """
    Prepares a test sequence from the given feature dictionary.
    Parameters:
        features: dict, the input feature data.
        min_sequence_length: int, minimum length of the sequence.
        feature_columns: list of str, columns to include in the sequence.
    Returns:
        Numpy array of the test sequence.
    """
    if feature_columns is None:
        raise ValueError("feature_columns must be provided.")
    sequence = np.column_stack([features[col][-min_sequence_length:] for col in feature_columns])
    return pad_sequences([sequence], maxlen=min_sequence_length, padding='post', dtype='float32')
