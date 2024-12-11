import os
import random

import pandas as pd


def generate_pseudo_testing_data(directory: str, directory_truth: str):
    """
    Generates pseudo-testing data by filtering rows based on the 'rul (months)' column.
    The filtered data is saved as CSV files in the specified directory.

    :param directory: Path to the directory where pseudo-testing data will be saved.
    :param directory_truth: Path to the directory containing the original ground truth CSV files.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)                                              # Create output directory if it does not exist

    csv_files = [f for f in os.listdir(directory_truth) if f.endswith('.csv')]

    for file_name in csv_files:
        df = pd.read_csv(os.path.join(directory_truth, file_name))
        ttf = int(df.iloc[0]['rul (months)'])                               # Time-to-failure (TTF) from the first row

        if ttf >= 6:
            random_integer = random.randint(1, ttf - 1)                  # Random cutoff point for filtering
            df = df[df['rul (months)'] >= random_integer]                   # Filter rows with 'rul (months)' greater than or equal to the random cutoff
            df.to_csv(os.path.join(directory, file_name), index=False)
        else:
            df = df[df['rul (months)'] > 0]                                 # Keep only rows with 'rul (months)' greater than 0
            df.to_csv(os.path.join(directory, file_name), index=False)


def generate_pseudo_testing_data_with_truth(directory: str, directory_student: str):
    """
    Generates pseudo-testing data with a corresponding solution file. The solution file contains
    item IDs, labels indicating whether the 'rul (months)' is less than or equal to 6, and the true RUL.

    :param directory: Path to the directory containing the pseudo-testing data.
    :param directory_student: Path to the directory where filtered data will be saved.
    """
    if not os.path.exists(directory_student):
        os.makedirs(directory_student)                                   # Create output directory if it does not exist

    solution = pd.DataFrame()                                            # Initialize an empty DataFrame for the solution

    for i in range(50):
        file_name = f'item_{i}.csv'
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)

        true_rul = df.iloc[-1]['rul (months)']                          # Get the true RUL from the last row

        # Append item details to the solution DataFrame
        solution = pd.concat([solution, pd.DataFrame([{
            'item_id': f'item_{i}',
            'label': 1 if true_rul <= 6 else 0,                         # Label as 1 if RUL <= 6, otherwise 0
            'true_rul': true_rul
        }])])

        df = df.drop(columns=['rul (months)'])                                   # Drop the 'rul (months)' column from the data
        df.to_csv(os.path.join(directory_student, file_name), index=False)

    solution.to_csv(os.path.join(directory, 'Solution.csv'), index=False)