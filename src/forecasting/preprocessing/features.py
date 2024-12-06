import pandas as pd
from statsmodels.tsa.seasonal import STL
from src.core import ParticleFilter


class FeatureAdder:
    """Class to add engineered features to the given DataFrame for time series data analysis.

    :param min_sequence_length: Minimum length of the sequence for rolling features.
    """

    def __init__(self, min_sequence_length):
        """Initializes the FeatureAdder class with the minimum sequence length."""
        self.min_sequence_length = min_sequence_length

    def add_features(self, df, particles_filtery):
        """
        Adds engineered features (rolling, static, shifts, ratios) to the DataFrame.

        :param df: Input DataFrame with the data to be transformed.
        :param particles_filtery: Flag to indicate if particle filtering should be applied.
        :return: DataFrame with added features.
        """

        def calculate_rolling_features(df, column, window_size):
            """
            Calculate rolling statistics (mean, std, max, min) for a given column.

            :param df: DataFrame containing the data.
            :param column: The column name for which the rolling features are calculated.
            :param window_size: The size of the rolling window.
            :return: Rolling mean, std, max, and min values.
            """
            rolling = df.groupby('item_id')[column].rolling(window=window_size, min_periods=1)
            rolling_mean = rolling.mean().reset_index(level=0, drop=True)
            rolling_std = rolling.std().reset_index(level=0, drop=True)
            rolling_max = rolling.max().reset_index(level=0, drop=True)
            rolling_min = rolling.min().reset_index(level=0, drop=True)
            return rolling_mean, rolling_std, rolling_max, rolling_min

        def calculate_static_features(df, column):
            """
            Calculate static statistics (mean, std, max, min) for a given column.

            :param df: DataFrame containing the data.
            :param column: The column name for which the static features are calculated.
            :return: Static mean, std, max, and min values.
            """
            group = df.groupby('item_id')[column]
            static_mean = group.transform('mean')
            static_std = group.transform('std')
            static_max = group.transform('max')
            static_min = group.transform('min')
            return static_mean, static_std, static_max, static_min

        def replace_nan_by_mean_and_one(df, columns):
            """
            Replace NaN values with the mean of the column, and remaining NaN with 1.

            :param df: DataFrame containing the columns to process.
            :param columns: List of column names to replace NaN values.
            :return: DataFrame with NaN values replaced.
            """
            for col in columns:
                df[col] = df[col].fillna(df[col].mean()).fillna(1)
            return df

        def particles_filtering(df):
            """
            Apply particle filtering to the DataFrame.

            :param df: DataFrame to apply particle filtering on.
            :return: DataFrame after particle filtering.
            """
            pf = ParticleFilter()
            df = pf.filter(df, beta0_range=(-1, 1), beta1_range=(-0.1, 0.1), beta2_range=(0.1, 1))
            return df

        # Particle filtering block
        if particles_filtery:
            # Drop existing particle filtering-related columns
            to_recall = [
                'length_filtered', 'beta0', 'beta1', 'beta2',
                'rolling_means_filtered', 'rolling_stds_filtered',
                'rolling_maxs_filtered', 'rolling_mins_filtered',
                'rolling_means_measured', 'rolling_stds_measured',
                'rolling_maxs_measured', 'rolling_mins_measured'
            ]
            df.drop(columns=[col for col in to_recall if col in df.columns], inplace=True)
            df = particles_filtering(df)
        else:
            # Drop only the rolling feature-related columns if no particle filtering
            to_recall = [
                'rolling_means_filtered', 'rolling_stds_filtered', 'rolling_maxs_filtered', 'rolling_mins_filtered',
                'rolling_means_measured', 'rolling_stds_measured', 'rolling_maxs_measured', 'rolling_mins_measured'
            ]
            df.drop(columns=[col for col in to_recall if col in df.columns], inplace=True)

        # Calculate rolling and static features for specific columns
        for col in ['time (months)', 'length_filtered']:
            rolling_mean, rolling_std, rolling_max, rolling_min = calculate_rolling_features(df, col,
                                                                                             self.min_sequence_length)
            static_mean, static_std, static_max, static_min = calculate_static_features(df, col)

            # Add the new features to the DataFrame
            df[f'rolling_mean_{col}'] = rolling_mean
            df[f'rolling_std_{col}'] = rolling_std
            df[f'rolling_max_{col}'] = rolling_max
            df[f'rolling_min_{col}'] = rolling_min

            df[f'static_mean_{col}'] = static_mean
            df[f'static_std_{col}'] = static_std
            df[f'static_max_{col}'] = static_max
            df[f'static_min_{col}'] = static_min

        # Replace NaN values in rolling and static feature columns
        rolling_columns = [f'rolling_mean_{col}', f'rolling_std_{col}', f'rolling_max_{col}', f'rolling_min_{col}',
                           f'static_mean_{col}', f'static_std_{col}', f'static_max_{col}', f'static_min_{col}']
        replace_nan_by_mean_and_one(df, rolling_columns)

        # Fill NaN for specific columns with zero
        df[['time (months)', 'length_measured', 'length_filtered']] = df[
            ['time (months)', 'length_measured', 'length_filtered']].fillna(0)

        def add_shifts_and_ratios(df, columns, max_shift=6):
            """
            Add shifted columns and ratios between consecutive shifts for given columns.

            :param df: DataFrame containing the data.
            :param columns: List of column names to process for shifting.
            :param max_shift: Maximum number of shifts to apply.
            :return: DataFrame with shifted columns and ratio columns added.
            """
            for col in columns:
                for shift in range(1, max_shift + 1):                           # Add shifted columns for each shift from 1 to max_shift
                    shifted_col = df.groupby('item_id')[col].shift(shift)
                    df[f'{col}_shift_{shift}'] = shifted_col

                for shift in range(1, max_shift):                                               # Calculate ratios between consecutive shifted columns
                    df[f'{col}_ratio_{shift}-{shift + 1}'] = df[f'{col}_shift_{shift}'] / (
                            df[f'{col}_shift_{shift + 1}'] + 1e-9)
            df.fillna(0, inplace=True)
            return df

        shift_columns = ['length_filtered']                 # Add shifts and ratios for 'length_filtered' column
        df = add_shifts_and_ratios(df, shift_columns)

        def decompose_time_series(df, time_col, value_col, period=12):
            """
            Decompose the time series into trend, seasonal, and residual components.

            :param df: DataFrame containing the time series data.
            :param time_col: The name of the column containing time data.
            :param value_col: The name of the column containing the values to decompose.
            :param period: The period for seasonal decomposition (default 12 months).
            :return: DataFrame with additional columns for trend, seasonal, and residual components.
            """
            if time_col not in df.columns or value_col not in df.columns:
                print(f"Columns '{time_col}' or '{value_col}' do not exist in the DataFrame.")
                return df

            df_copy = df.copy()

            start_date = pd.Timestamp('2024-01-01')                                                             # Set an init start date and adjust time column
            df_copy[time_col] = df_copy[time_col].apply(lambda x: start_date + pd.DateOffset(months=int(x)))

            df_copy = df_copy.sort_values(by=time_col)                      # Sort the DataFrame and drop rows with missing time or value data
            df_copy = df_copy.dropna(subset=[time_col, value_col])

            try:
                stl = STL(df_copy[value_col], period=period)    # Apply STL decomposition
                result = stl.fit()
            except Exception as e:
                print(f"Error during STL decomposition: {e}")
                return df

            df['Trend'] = result.trend                          # Add decomposition results to the original DataFrame
            df['Seasonal'] = result.seasonal
            df['Residual'] = result.resid

            return df

        df = decompose_time_series(df, 'time (months)', 'length_filtered')  # Decompose time series using 'time (months)' and 'length_filtered'

        df.sort_values(by=["item_id", "time (months)"], ascending=[True, True], inplace=True) # Sort the final DataFrame

        return df
