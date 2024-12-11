from typing import Optional, List, Tuple, Dict

import os
import re
import numpy as np
import pandas as pd
from sksurv.util import Surv

from src.core.generate_data import generate_pseudo_testing_data, generate_pseudo_testing_data_with_truth
from src.core.visualizer import DataVisualizer

def prepare_train_data(
    df: pd.DataFrame,
    columns_to_include: List[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepares the data for training and prediction.
    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        columns_to_include (List[str], optional): List of column names to include in the training data.
            If None, all columns will be included.
    Returns:
        Tuple[pd.DataFrame, np.ndarray]:
            - x_train (pd.DataFrame): DataFrame with selected columns for training.
            - y_train (np.ndarray): Array representing the target values (e.g., labels for classification).
    """
    df = df.sort_values(by=['item_id', 'time (months)']).reset_index(drop=True)     # Sort and reset index

    df['label'] = df['label'].astype(bool)                                          # Convert and clean columns
    df['time (months)'] = pd.to_numeric(df['time (months)'], errors='coerce')
    df.dropna(subset=['time (months)', 'label'], inplace=True)

    x_train = df[columns_to_include]        # Select columns for training

    try:                                                                                    # Create survival object
        y_train = Surv.from_dataframe('label', 'time (months)', x_train)
    except ValueError as e:
        print(f"Error creating survival object: {e}")
        raise e

    return x_train, y_train


def generate_validation_sets(n_validation_sets: int) -> Dict[str, pd.DataFrame]:
    """
    Generates multiple validation datasets by processing pseudo-testing data files and
    combining them into DataFrames.
    Args:
        n_validation_sets (int): The number of validation sets to generate.
    Returns:
        Dict[str, pd.DataFrame]:
            A dictionary where the keys are the identifiers for each validation set (e.g., 'validation_set_1'),
            and the values are the corresponding DataFrames created by concatenating the pseudo-testing data files.
    """
    validation_data = {}

    for i in range(n_validation_sets):

        generate_pseudo_testing_data(
            directory='data/input/training_data/pseudo_testing_data_with_truth',
            directory_truth='data/input/training_data/degradation_data'
        )
        generate_pseudo_testing_data_with_truth(
            directory='data/input/training_data/pseudo_testing_data_with_truth',
            directory_student='data/input/training_data/pseudo_testing_data'
        )
        pseudo_testing_data_with_truth_path = './data/input/training_data/pseudo_testing_data_with_truth'
        pseudo_testing_data_with_truth_dfs = []

        for filename in os.listdir(pseudo_testing_data_with_truth_path):
            if filename.endswith('.csv'):
                match = re.search(r'(\d+)', filename)
                if match:
                    item_id = int(match.group(1))
                    file_path = os.path.join(pseudo_testing_data_with_truth_path, filename)
                    df = pd.read_csv(file_path)
                    df['item_id'] = item_id
                    pseudo_testing_data_with_truth_dfs.append(df)

        validation_df = pd.concat(pseudo_testing_data_with_truth_dfs, ignore_index=True)

        tools = DataVisualizer()
        validation_df = tools.preprocessing(validation_df)

        validation_data[f'validation_set_{i + 1}'] = validation_df

    return validation_data


def prepare_validation_sets(
        n_sets: int = None,
        reference_df: Optional[pd.DataFrame] = None,
        columns_to_include: Optional[List[str]] = None
    ) -> Dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """
    Prepares multiple validation datasets by processing raw data into formatted training
    and prediction data for survival analysis.

    Args:
        n_sets (int): The number of validation sets to generate.
        reference_df (pd.DataFrame, optional): A reference DataFrame used to align the columns.
            If provided, columns in each validation set will be reindexed to match this reference.
        columns_to_include (List[str], optional): A list of column names to include in the feature set
            (x_val) for training. If None, all columns will be included.

    Returns:
        Dict[str, tuple[pd.DataFrame, np.ndarray]]:
            A dictionary where the keys are the identifiers for each validation set (e.g., 'validation_set_1'),
            and the values are tuples containing:
            - x_val (pd.DataFrame): DataFrame with selected columns for training.
            - y_val (np.ndarray): Array of target values (labels and time for survival analysis).
    """
    validation_data = {}
    data_generated = generate_validation_sets(n_sets)  # Generate validation sets

    for key, df in data_generated.items():

        if 'true_rul' in df.columns:
            df.drop(columns=['true_rul'], inplace=True)

        df = df.sort_values(by=['item_id', 'time (months)']).reset_index(drop=True)

        if reference_df is not None:
            df = df.reindex(columns=reference_df.columns, fill_value=0)

        df['label'] = df['label'].astype(bool)
        df['time (months)'] = pd.to_numeric(df['time (months)'], errors='coerce')
        df.dropna(subset=['time (months)', 'label'], inplace=True)

        x_val = df[columns_to_include]

        try:
            y_val = Surv.from_dataframe('label', 'time (months)', x_val)
        except ValueError as e:
            print(f"Error creating survival object for {key}: {e}")
            raise e

        validation_data[key] = (x_val, y_val)

    return validation_data