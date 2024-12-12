import streamlit as st
from typing import Optional, List, Tuple, Dict
import os
import re
import numpy as np
import pandas as pd

from src.core.generate_data import generate_pseudo_testing_data, generate_pseudo_testing_data_with_truth
from src.core.utils import dataframing_data
from src.forecasting import clean_data, FeatureAdder

from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample


def normalize_data(df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
    """
    Normalise les variables numériques dans le DataFrame.
    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        numerical_columns (List[str]): Liste des colonnes à normaliser.
    Returns:
        pd.DataFrame: DataFrame avec les variables normalisées.
    """
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def balance_classes(x: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Équilibre les classes en utilisant SMOTE (suréchantillonnage).
    Args:
        x (pd.DataFrame): Features.
        y (np.ndarray): Cibles (labels).
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Features et cibles équilibrées.
    """
    smote = SMOTE(random_state=42)
    #original_indices = x.index
    x_balanced, y_balanced = smote.fit_resample(x, y)
    #x_balanced.index = original_indices.append(pd.Index(range(len(x_balanced) - len(original_indices))))
    return x_balanced, y_balanced


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data to clean and add features.
    Args:
        df (pd.DataFrame): Input DataFrame containing raw data.
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    feature_adder = FeatureAdder(min_sequence_length=2)
    df = df.rename(columns={'crack length (arbitary unit)': 'length_measured'})
    df = df.sort_values(by=['item_id', 'time (months)'])
    df['item_id'] = df['item_id'].astype(int)
    df = feature_adder.add_features(clean_data(df), particles_filtery=True, verbose=False)
    df = normalize_data(
        df=df,
        numerical_columns=['length_measured', 'length_filtered']
    )
    return df


def create_survival_data(
        df: pd.DataFrame,
        columns_to_include: List[str],
        balance_classes_flag: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create survival data (features and survival objects) from a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame containing data.
        columns_to_include (List[str]): Columns to include in the feature set.
        balance_classes_flag (bool, optional): Flag to balance classes.
    Returns:
        Tuple[pd.DataFrame, np.ndarray]:
            - Features DataFrame (x_).
            - Survival object array (y_).
    """
    x_ = df[columns_to_include]
    y_ = df['label']

    if balance_classes_flag:
        x_, y_ = balance_classes(x_, y_)

    try:
        y_ = Surv.from_dataframe('label', 'time (months)', x_)
    except ValueError as e:
        print(f"Error creating survival object: {e}")
        raise e

    return x_, y_


def prepare_train_and_test_sets(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    columns_to_include: Optional[List[str]] = None,
    balance_classes_flag: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare training and testing data for survival analysis.
    Args:
        df (pd.DataFrame): Input data.
        reference_df (pd.DataFrame, optional): Reference DataFrame for column alignment.
        columns_to_include (List[str], optional): Columns to include in the output DataFrame.
        balance_classes_flag (bool, optional): Flag to balance classes.
    Returns:
        Tuple[pd.DataFrame, np.ndarray]:
            - x_: Features DataFrame.
            - y_: Survival object array.
    """
    df = df.sort_values(by=['item_id', 'time (months)']).reset_index(drop=True)

    if reference_df is not None:
        df = df.reindex(columns=reference_df.columns, fill_value=0)

    df['label'] = df['label'].astype(bool)
    df['time (months)'] = pd.to_numeric(df['time (months)'], errors='coerce')
    df.dropna(subset=['time (months)', 'label'], inplace=True)

    return create_survival_data(df, columns_to_include, balance_classes_flag)


def generate_n_validation_sets(n_validation_sets: int) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple validation datasets.
    Args:
        n_validation_sets (int): Number of validation sets to generate.
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of validation sets.
    """
    validation_data = {}
    pseudo_testing_path = './data/input/training_data/pseudo_testing_data_with_truth'

    for i in range(n_validation_sets):
        generate_pseudo_testing_data(
            directory='data/input/training_data/pseudo_testing_data_with_truth',
            directory_truth='data/input/training_data/degradation_data'
        )
        generate_pseudo_testing_data_with_truth(
            directory='data/input/training_data/pseudo_testing_data_with_truth',
            directory_student='data/input/training_data/pseudo_testing_data'
        )

        dfs = []
        for filename in os.listdir(pseudo_testing_path):
            if filename.endswith('.csv'):
                match = re.search(r'\d+', filename)
                if match:
                    item_id = int(match.group(0))
                    file_path = os.path.join(pseudo_testing_path, filename)
                    df = pd.read_csv(file_path)
                    df['item_id'] = item_id
                    dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        # Load solution data
        solution_data_path = './data/input/training_data/pseudo_testing_data_with_truth/Solution.csv'
        solution_data = pd.read_csv(solution_data_path)

        # Merge combined df with solution data
        df = combined_df.copy()
        solution_data['item_id'] = solution_data['item_id'].str.extract(r'(\d+)').astype(int)
        df = pd.merge(df, solution_data, on='item_id', how='left')
        df['item_id'] = df['item_id'].astype(int)

        validation_data[f'validation_set_{i + 1}'] = process_data(df)

    return validation_data


def prepare_validation_sets(
    n_sets: int,
    reference_df: Optional[pd.DataFrame] = None,
    columns_to_include: Optional[List[str]] = None,
    balance_classes_flag: bool = False
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Prepare validation datasets for survival analysis.
    Args:
        n_sets (int): Number of validation sets to prepare.
        reference_df (pd.DataFrame, optional): Reference DataFrame for column alignment.
        columns_to_include (List[str], optional): Columns to include in feature sets.
        balance_classes_flag (bool, optional): Flag to balance classes.
    Returns:
        Dict[str, Tuple[pd.DataFrame, np.ndarray]]: Dictionary of validation sets with features and survival objects.
    """
    validation_data = {}
    truth_data = {}
    generated_data = generate_n_validation_sets(n_sets)

    for key, df in generated_data.items():
        df = df.sort_values(by=['item_id', 'time (months)']).reset_index(drop=True)

        if 'true_rul' in df.columns:
            truth_data[key] = df.copy()  # Save the dataframe with 'true_rul' for evaluation

        df.drop(columns=['true_rul'], inplace=True, errors='ignore')

        if reference_df is not None:
            df = df.reindex(columns=reference_df.columns, fill_value=0)

        df['label'] = df['label'].astype(bool)
        df['time (months)'] = pd.to_numeric(df['time (months)'], errors='coerce')
        df.dropna(subset=['time (months)', 'label'], inplace=True)

        validation_data[key] = create_survival_data(df, columns_to_include, balance_classes_flag)

    return validation_data, truth_data


def prepare_data(selected_variables: List[str], n_validation_sets: int = 10, balance_classes_flag: bool = False) -> Dict[str, object]:
    """
    Prepare datasets for training, validation, and testing.
    Args:
        selected_variables (List[str]): Columns to include in the datasets.
        n_validation_sets (int): Number of validation sets to generate.
        balance_classes_flag (bool): Flag to balance classes.
    Returns:
        Dict[str, object]: Dictionary containing prepared datasets.
    """
    # Loading from source
    dataframes = dataframing_data()
    train_df, test_df = process_data(dataframes['train']), process_data(dataframes['test'])

    # Prepare training data
    x_train, y_train = prepare_train_and_test_sets(
        train_df, columns_to_include=selected_variables,
        balance_classes_flag=balance_classes_flag
    )
    print(f"training_set: x_train shape {x_train.shape}, y_train shape {y_train.shape}")

    # Prepare validation data
    validation_data, truth_data = prepare_validation_sets(
        n_sets=n_validation_sets,
        reference_df=train_df,
        columns_to_include=selected_variables,
        balance_classes_flag=balance_classes_flag
    )
    for key, (x_val, y_val) in validation_data.items():
        print(f"{key}: x_val shape {x_val.shape}, y_val shape {y_val.shape}")

    # Prepare testing data
    x_test, _ = prepare_train_and_test_sets(
        test_df, reference_df=train_df, columns_to_include=selected_variables
    )
    print(f"test_set: x_test shape {x_test.shape}, _ shape {_.shape}")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "validation_data": validation_data,
        "truth_data": truth_data,
        "x_test": x_test
    }