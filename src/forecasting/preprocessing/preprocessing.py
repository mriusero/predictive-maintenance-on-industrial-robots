import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def clean_data(df):
    """
    Cleans the input DataFrame by handling missing values, transforming columns,
    and ensuring the correct format for specific data attributes.

    :param df: DataFrame to be cleaned.
    :return: Cleaned DataFrame with missing values handled, columns transformed,
             and necessary adjustments made.
    :raises: ValueError if required columns are missing from the DataFrame.
    """

    def complete_empty_item_data(group):
        """
        Completes missing data for a specific group (identified by 'item_id') by adding default
        values for missing columns and transforming certain columns.

        :param group: A group of rows in the DataFrame corresponding to a unique 'item_id'.
        :return: The modified group with missing data filled and transformations applied.
        """
        # Add missing columns with default values (0)
        group = group.assign(
            source=group.get('source', pd.Series(0, index=group.index)),
            label=group.get('label', pd.Series(0, index=group.index))
        )
        group['true_rul'] = group.get('true_rul', group['label']).astype(int)   # Update or create 'true_rul' column with integer values

        # Transform 'Failure mode' and related columns if they exist
        if 'Failure mode' in group.columns:
            group['failure_month'] = (group['Time to failure (months)'] == group['time (months)']).astype(int)
            group.loc[
                group['failure_month'] != 1, ['Infant mortality', 'Control board failure', 'Fatigue crack']] = False

            # Fill 'Failure mode' and 'Time to failure (months)' if both are missing
            if group['Failure mode'].isnull().all() and group['Time to failure (months)'].isnull().all():
                group['Failure mode'] = 'Fatigue crack'
                group['Time to failure (months)'] = group['time (months)'].max()

        # Convert 'Time to failure (months)' to integer if it exists
        if 'Time to failure (months)' in group.columns:
            group['Time to failure (months)'] = group['Time to failure (months)'].fillna(0).astype(int)

        # Calculate 'rul (months)' if the column exists
        if 'rul (months)' in group.columns:
            max_time = group['time (months)'].max()
            group['rul (months)'] = max_time - group['time (months)'] + 1

        return group

    # Convert 'Failure mode' to one-hot encoded columns if it exists
    if 'Failure mode' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['Failure mode'], prefix=None)], axis=1)


    df = df.groupby('item_id', group_keys=False).apply(complete_empty_item_data)        # Apply the cleaning function for each 'item_id' group
    df.dropna(inplace=True)                                                                 # Drop rows with any missing values

    # Add missing columns with default values (0)
    all_possible_columns = ['source', 'item_id', 'time (months)', 'length_measured', 'label',
                            'Failure mode', 'Infant mortality', 'Control board failure', 'Fatigue crack',
                            'Time to failure (months)', 'rul (months)', 'true_rul', 'failure_month']
    missing_columns = set(all_possible_columns) - set(df.columns)
    for col in missing_columns:
        df[col] = 0

    # Reorganize columns: common ones first, followed by remaining ones sorted alphabetically
    common_columns = ['source', 'item_id', 'time (months)', 'length_measured', 'label',
                      'Failure mode', 'Infant mortality', 'Control board failure', 'Fatigue crack',
                      'Time to failure (months)', 'rul (months)', 'true_rul', 'failure_month']
    df = df[common_columns + sorted(set(df.columns) - set(common_columns))]

    # Create and set a unique index combining 'item_id' and 'time (months)'
    df['unique_index'] = df['item_id'].astype(str) + "_&_mth_" + df['time (months)'].astype(str)
    df.set_index('unique_index', inplace=True)

    return df


def standardize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Standardizes the specified columns in the DataFrame by removing the mean and scaling to unit variance.

    :param df: DataFrame containing the data to be standardized.
    :param columns: List of column names to standardize.
    :return: DataFrame with the specified columns standardized.
    :raises: ValueError if any column is missing from the DataFrame.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])  # Standardize the specified columns
    return df


def normalize_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Normalizes the specified columns in the DataFrame to a range between 0 and 1.

    :param df: DataFrame containing the data to be normalized.
    :param columns: List of column names to normalize.
    :return: DataFrame with the specified columns normalized.
    :raises: ValueError if any column is missing from the DataFrame.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])  # Normalize the specified columns
    return df