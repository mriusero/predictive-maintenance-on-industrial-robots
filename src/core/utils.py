import gc
import os
import re
from datetime import datetime

import pandas as pd


def load_data():
    """
    Loads multiple datasets from various CSV files and merges them into a unified dataset.
    :return: str: A message indicating that the data has been successfully generated.
    """
    # Load failure data
    failure_data_path = './data/input/training_data/failure_data.csv'
    failure_data = pd.read_csv(failure_data_path)

    # Load degradation data
    degradation_data_path = './data/input/training_data/degradation_data'
    degradation_dfs = []
    for filename in os.listdir(degradation_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(degradation_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            degradation_dfs.append(df)
    degradation_data = pd.concat(degradation_dfs, ignore_index=False)

    # Load pseudo testing data
    pseudo_testing_data_path = './data/input/training_data/pseudo_testing_data'
    pseudo_testing_dfs = []
    for filename in os.listdir(pseudo_testing_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(pseudo_testing_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            pseudo_testing_dfs.append(df)
    pseudo_testing_data = pd.concat(pseudo_testing_dfs, ignore_index=False)

    # Load pseudo testing data with truth
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
    pseudo_testing_data_with_truth = pd.concat(pseudo_testing_data_with_truth_dfs, ignore_index=False)

    # Load solution data
    solution_data_path = './data/input/training_data/pseudo_testing_data_with_truth/Solution.csv'
    solution_data = pd.read_csv(solution_data_path)

    # Load testing data
    testing_data_path = './data/input/testing_data/phase1'
    testing_dfs = []
    for filename in os.listdir(testing_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[2].split('.')[0])
            file_path = os.path.join(testing_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            testing_dfs.append(df)
    testing_data = pd.concat(testing_dfs, ignore_index=False)

    # Consolidate all datasets into a dictionary
    data = {
        'failure_data': failure_data,
        'degradation_data': degradation_data,
        'pseudo_testing_data': pseudo_testing_data,
        'pseudo_testing_data_with_truth': pseudo_testing_data_with_truth,
        'solution_data': solution_data,
        'testing_data': testing_data
    }

    return merge_data(data)


def merge_data(training_data):
    """
    Merges various datasets into a final dataset and saves them to CSV files.
    :param training_data: dict: Dictionary containing the datasets to merge.
    :return: str: A message indicating the successful generation of new data.
    """
    # Merge degradation and failure data
    df1 = pd.merge(training_data['degradation_data'], training_data['failure_data'], on='item_id', how='left')
    df1['label'] = (df1['rul (months)'] <= 6).astype(int)  # Label items with remaining useful life <= 6 months
    df1 = df1.sort_values(by=["item_id", "time (months)"], ascending=[True, True])

    # Prepare pseudo testing data
    df2 = training_data['pseudo_testing_data'].copy()
    df2['item_id'] = df2['item_id'].astype(int)

    # Merge pseudo testing data with truth and solution data
    df3 = training_data['pseudo_testing_data_with_truth'].copy()
    training_data['solution_data']['item_id'] = training_data['solution_data']['item_id'].str.extract(r'(\d+)').astype(int)
    df3 = pd.merge(df3, training_data['solution_data'], on='item_id', how='left')
    df3['item_id'] = df3['item_id'].astype(int)

    # Prepare testing data
    df4 = training_data['testing_data'].copy()
    df4['item_id'] = df4['item_id'].astype(int)

    # Save merged data to CSV
    df1.to_csv('./data/output/training/training_data.csv', index=False)
    df2.to_csv('./data/output/pseudo_testing/pseudo_testing_data.csv', index=False)
    df3.to_csv('./data/output/pseudo_testing/pseudo_testing_data_with_truth.csv', index=False)
    df4.to_csv('./data/output/testing/testing_data_phase1.csv', index=False)

    # Generate success message
    update_message = 'New data generated successfully!'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Clean up and clear terminal
    gc.collect()
    os.system('clear')

    return f"{update_message} [{timestamp}]"


def dataframing_data():
    """
    Reads and loads all processed datasets into DataFrames.
    :return: dict: A dictionary containing all the loaded DataFrames.
    """
    paths = {
        'train': './data/output/training/training_data.csv',
        'pseudo_test': './data/output/pseudo_testing/pseudo_testing_data.csv',
        'pseudo_test_with_truth': './data/output/pseudo_testing/pseudo_testing_data_with_truth.csv',
        'test': './data/output/testing/testing_data_phase1.csv'
    }
    dataframes = {
        'train': pd.read_csv(paths['train']),
        'pseudo_test': pd.read_csv(paths['pseudo_test']),
        'pseudo_test_with_truth': pd.read_csv(paths['pseudo_test_with_truth']),
        'test': pd.read_csv(paths['test'])
    }
    return dataframes


def load_failures():
    """
    Loads failure modes for all items from the training dataset.
    :return: pd.DataFrame: DataFrame containing the item_id and its corresponding failure mode.
    :raises ValueError: If 'item_id' or 'Failure mode' columns are missing from the data.
    """
    df = pd.read_csv('./data/output/training/training_data.csv')
    if 'item_id' not in df.columns or 'Failure mode' not in df.columns:
        raise ValueError("The columns 'item_id' or 'Failure mode' are missing in the DataFrame.")
    df = df.groupby('item_id')['Failure mode'].first().reset_index()
    failures_df = df.rename(columns={'Failure mode': 'Failure mode'})
    return failures_df


def display_variable_types(df):
    """
    Displays the variable types of a given DataFrame in a user-friendly format.

    :param df: pd.DataFrame: The DataFrame to analyze.
    :return: pd.DataFrame: DataFrame with variable names and their types.
    """
    def identify_variable_type(series):
        """
        Identifies the type of a variable in a pandas Series.
        """
        if pd.api.types.is_numeric_dtype(series):
            unique_values = series.nunique()
            total_values = len(series)
            if pd.api.types.is_float_dtype(series) or unique_values > 20 and unique_values / total_values > 0.05:
                return 'continuous'
            elif pd.api.types.is_integer_dtype(series) or unique_values <= 20:
                return 'discrete'
        else:
            unique_values = series.nunique()
            total_values = len(series)
            if unique_values / total_values < 0.5:
                return 'categorical'
        return 'unknown'

    results = {'Variable': [], 'Type': []}
    for col in df.columns:
        var_type = identify_variable_type(df[col])
        results['Variable'].append(col)
        results['Type'].append(var_type)

    return pd.DataFrame(results)