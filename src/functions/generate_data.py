import os
import pandas as pd
import random

def generate_pseudo_testing_data(directory: str, directory_truth: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    csv_files = [f for f in os.listdir(directory_truth) if f.endswith('.csv')]

    for i, file_name in enumerate(csv_files):
        df = pd.read_csv(os.path.join(directory_truth, file_name))
        ttf = df.iloc[0]['rul (months)']

        ttf = int(ttf)

        if ttf >= 6:
            random_integer = random.randint(1, ttf-1)

            df = df[df['rul (months)'] >= random_integer]
            df.to_csv(os.path.join(directory, file_name), index=False)
        else:
            df = df[df['rul (months)'] > 0]
            df.to_csv(os.path.join(directory, file_name), index=False)

def generate_pseudo_testing_data_with_truth(directory: str, directory_student: str):
    if not os.path.exists(directory_student):
        os.makedirs(directory_student)

    solution = pd.DataFrame()

    for i in range(50):
        file_name = f'item_{i}.csv'
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)

        true_rul = df.iloc[-1]['rul (months)']

        solution = pd.concat([solution, pd.DataFrame([{
            'item_id': f'item_{i}',
            'label': 1 if true_rul <= 6 else 0,
            'true_rul': true_rul
        }])])

        df = df.drop(columns=['rul (months)'])
        df.to_csv(os.path.join(directory_student, file_name), index=False)

    solution.to_csv(os.path.join(directory, 'Solution.csv'), index=False)
