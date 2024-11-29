import pandas as pd
import os
import re
import glob
from datetime import datetime
import gc

def load_data():

    failure_data_path = './data/input/training_data/failure_data.csv'           #Failure data
    failure_data = pd.read_csv(failure_data_path)

    degradation_data_path = './data/input/training_data/degradation_data'       #Degradation data
    degradation_dfs = []
    for filename in os.listdir(degradation_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(degradation_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            degradation_dfs.append(df)
    degradation_data = pd.concat(degradation_dfs, ignore_index=False)

    pseudo_testing_data_path = './data/input/training_data/pseudo_testing_data'     #Pseudo testing
    pseudo_testing_dfs = []
    for filename in os.listdir(pseudo_testing_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[1].split('.')[0])
            file_path = os.path.join(pseudo_testing_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            pseudo_testing_dfs.append(df)
    pseudo_testing_data = pd.concat(pseudo_testing_dfs, ignore_index=False)

    pseudo_testing_data_with_truth_path = './data/input/training_data/pseudo_testing_data_with_truth'  # Pseudo testing with truth
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
    pseudo_testing_data_with_truth = pd.concat(pseudo_testing_dfs, ignore_index=False)

    solution_data_path = './data/input/training_data/pseudo_testing_data_with_truth/Solution.csv'   # Solution
    solution_data = pd.read_csv(solution_data_path)

    testing_data_path = './data/input/testing_data/phase1'  # Testing data
    testing_dfs = []
    for filename in os.listdir(testing_data_path):
        if filename.endswith('.csv'):
            item_id = int(filename.split('_')[2].split('.')[0])
            file_path = os.path.join(testing_data_path, filename)
            df = pd.read_csv(file_path)
            df['item_id'] = item_id
            testing_dfs.append(df)
    testing_data = pd.concat(testing_dfs, ignore_index=False)

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

    df1 = pd.merge(training_data['degradation_data'], training_data['failure_data'], on='item_id', how='left')
    df1['label'] = (df1['rul (months)'] <= 6).astype(int)
    df1 = df1.sort_values(by=["item_id", "time (months)"], ascending=[True, True])


    df2 = training_data['pseudo_testing_data'].copy()
    df2['item_id'] = df2['item_id'].astype(int)

    df3 = training_data['pseudo_testing_data_with_truth'].copy()

    training_data['solution_data']['item_id'] = training_data['solution_data']['item_id'].str.extract(r'(\d+)').astype(int)
    df3 = pd.merge(df3, training_data['solution_data'], on='item_id', how='left')
    df3['item_id'] = df3['item_id'].astype(int)

    df4 = training_data['testing_data'].copy()
    df4['item_id'] = df4['item_id'].astype(int)


    df1.to_csv('./data/output/training/training_data.csv', index=False)
    df2.to_csv('./data/output/pseudo_testing/pseudo_testing_data.csv', index=False)
    df3.to_csv('./data/output/pseudo_testing/pseudo_testing_data_with_truth.csv', index=False)
    df4.to_csv('./data/output/testing/testing_data_phase1.csv', index=False)

    update_message = 'New data generated successfully !'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    gc.collect()
    os.system('clear')

    return f"{update_message} [{timestamp}]"

def dataframing_data():
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

    df = pd.read_csv('./data/output/training/training_data.csv')
    if 'item_id' not in df.columns or 'Failure mode' not in df.columns:
        raise ValueError("Les colonnes 'item_id' ou 'Failure mode' sont manquantes dans le DataFrame.")
    df = df.groupby('item_id')['Failure mode'].first().reset_index()
    failures_df = df.rename(columns={'Failure mode': 'Failure mode'})
    return failures_df




def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    outliers_list = []
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outliers = df[outlier_mask].copy()
        if not outliers.empty:
            outliers['outlier_column'] = column
            outliers['outlier_type'] = 'High'  # Possible extension: could also mark 'Low' for lower outliers
            outliers_list.append(outliers)

    if outliers_list:
        all_outliers_df = pd.concat(outliers_list)
        return all_outliers_df
    else:
        return pd.DataFrame(columns=df.columns.tolist() + ['outlier_column', 'outlier_type'])

def combine_submissions_for_scenario(folder_path):

    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs)
    final_df = combined_df.groupby('item_index').agg({'predicted_rul': 'max'}).reset_index()
    output_file = os.path.join(folder_path, 'Submission.csv')
    final_df.to_csv(output_file, index=False)


def display_variable_types(df):
    """
    Affiche dans Streamlit les types de variables d'un DataFrame donné.

    Paramètres:
    df (pd.DataFrame): Le DataFrame à analyser.
    """

    def identify_variable_type(series):
        """
        Identifie le type d'une variable dans une série pandas.
        """
        # Vérifier si la série contient des valeurs numériques
        if pd.api.types.is_numeric_dtype(series):
            unique_values = series.nunique()
            total_values = len(series)

            # Critère pour une variable continue
            if pd.api.types.is_float_dtype(series) or unique_values > 20 and unique_values / total_values > 0.05:
                return 'continue'
            # Critère pour une variable discrète
            elif pd.api.types.is_integer_dtype(series) or unique_values <= 20:
                return 'discrete'
        else:
            # Critère pour une variable catégorielle
            unique_values = series.nunique()
            total_values = len(series)
            if unique_values / total_values < 0.5:  # Ajuster le seuil selon vos besoins
                return 'categorical'

        # Si aucun des critères n'est satisfait
        return 'unknown'

    # Identification des types de variables pour chaque colonne du DataFrame
    results = {'Variable': [], 'Type': []}

    for col in df.columns:
        var_type = identify_variable_type(df[col])
        results['Variable'].append(col)
        results['Type'].append(var_type)

    # Convertir les résultats en DataFrame pour un affichage propre
    return pd.DataFrame(results)


def compare_dataframes(df1, df2):
    """
    Compare deux DataFrames et affiche les différences de colonnes et de valeurs.
    """
    # Comparaison des colonnes
    cols_diff = set(df1.columns).symmetric_difference(set(df2.columns))
    if cols_diff:
        print("Colonnes différentes :")
        print(cols_diff)
    else:
        print("Les deux DataFrames ont les mêmes colonnes.")

    # Comparaison des valeurs
    diff = df1.compare(df2, keep_shape=True, keep_equal=True)
    if diff.empty:
        print("Les deux DataFrames sont identiques au niveau des valeurs.")
    else:
        print("Différences de valeurs :")
        print(diff)
