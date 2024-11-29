# preprocessing.py

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd


def clean_data(df):
    """Nettoie les données en supprimant les valeurs manquantes et en transformant la colonne 'Failure mode'."""

    # Complète les données manquantes et transforme certaines colonnes
    def complete_empty_item_data(group):
        # Ajouter des colonnes manquantes avec des valeurs par défaut
        group = group.assign(
            source=group.get('source', pd.Series(0, index=group.index)),
            label=group.get('label', pd.Series(0, index=group.index))
        )

        # Mettre à jour ou définir la colonne 'true_rul'
        group['true_rul'] = group.get('true_rul', group['label']).astype(int)

        # Transformer 'Failure mode' et les colonnes associées
        if 'Failure mode' in group.columns:
            group['failure_month'] = (group['Time to failure (months)'] == group['time (months)']).astype(int)
            group.loc[
                group['failure_month'] != 1, ['Infant mortality', 'Control board failure', 'Fatigue crack']] = False

            if group['Failure mode'].isnull().all() and group['Time to failure (months)'].isnull().all():
                group['Failure mode'] = 'Fatigue crack'
                group['Time to failure (months)'] = group['time (months)'].max()

        # Convertir la colonne 'Time to failure (months)' en entier si elle existe
        if 'Time to failure (months)' in group.columns:
            group['Time to failure (months)'] = group['Time to failure (months)'].fillna(0).astype(int)

        # Calculer 'rul (months)' si la colonne existe
        if 'rul (months)' in group.columns:
            max_time = group['time (months)'].max()
            group['rul (months)'] = max_time - group['time (months)'] + 1

        return group

    # Convertir la colonne 'Failure mode' en variables indicatrices (1 et 0) si elle existe
    if 'Failure mode' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['Failure mode'], prefix=None)], axis=1)

    # Appliquer la fonction de nettoyage au niveau de chaque groupe 'item_id'
    df = df.groupby('item_id', group_keys=False).apply(complete_empty_item_data)

    # Supprimer les valeurs manquantes
    df.dropna(inplace=True)

    # Ajouter des colonnes manquantes avec des valeurs par défaut (0)
    all_possible_columns = ['source', 'item_id', 'time (months)', 'length_measured', 'label',
                            'Failure mode', 'Infant mortality', 'Control board failure', 'Fatigue crack',
                            'Time to failure (months)', 'rul (months)', 'true_rul', 'failure_month']
    missing_columns = set(all_possible_columns) - set(df.columns)
    for col in missing_columns:
        df[col] = 0

    # Réorganiser les colonnes selon les colonnes communes suivies des colonnes restantes triées
    common_columns = ['source', 'item_id', 'time (months)', 'length_measured', 'label',
                      'Failure mode', 'Infant mortality', 'Control board failure', 'Fatigue crack',
                      'Time to failure (months)', 'rul (months)', 'true_rul', 'failure_month']
    df = df[common_columns + sorted(set(df.columns) - set(common_columns))]

    # Créer et définir l'index unique
    df['unique_index'] = df['item_id'].astype(str) + "_&_mth_" + df['time (months)'].astype(str)
    df.set_index('unique_index', inplace=True)

    return df

def standardize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Standardise les valeurs des colonnes spécifiées."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def normalize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalise les valeurs des colonnes spécifiées."""
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


