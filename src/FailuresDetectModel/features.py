import pandas as pd
from statsmodels.tsa.seasonal import STL
from src.functions import ParticleFilter

class FeatureAdder:
    def __init__(self, min_sequence_length):
        self.min_sequence_length = min_sequence_length

    def add_features(self, df, particles_filtery):
        """Feature Engineering"""

        def calculate_rolling_features(df, column, window_size):
            # Calcul des statistiques de fenêtres mobiles
            rolling = df.groupby('item_id')[column].rolling(window=window_size, min_periods=1)
            rolling_mean = rolling.mean().reset_index(level=0, drop=True)
            rolling_std = rolling.std().reset_index(level=0, drop=True)
            rolling_max = rolling.max().reset_index(level=0, drop=True)
            rolling_min = rolling.min().reset_index(level=0, drop=True)
            return rolling_mean, rolling_std, rolling_max, rolling_min

        def calculate_static_features(df, column):
            # Calcul des statistiques globales
            group = df.groupby('item_id')[column]
            static_mean = group.transform('mean')
            static_std = group.transform('std')
            static_max = group.transform('max')
            static_min = group.transform('min')
            return static_mean, static_std, static_max, static_min

        def replace_nan_by_mean_and_one(df, columns):
            # Remplace NaN par la moyenne, puis par 1 pour les 0 restants
            for col in columns:
                df[col] = df[col].fillna(df[col].mean()).fillna(1)
            return df

        def particles_filtering(df):
            pf = ParticleFilter()
            df = pf.filter(df, beta0_range=(-1, 1), beta1_range=(-0.1, 0.1), beta2_range=(0.1, 1))
            return df

        if particles_filtery:
            # Suppression des colonnes si elles existent
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
            # Suppression des colonnes si elles existent
            to_recall = [
                'rolling_means_filtered', 'rolling_stds_filtered', 'rolling_maxs_filtered', 'rolling_mins_filtered',
                'rolling_means_measured', 'rolling_stds_measured', 'rolling_maxs_measured', 'rolling_mins_measured'
            ]
            df.drop(columns=[col for col in to_recall if col in df.columns], inplace=True)

        # Calcul des rolling features et des static features
        for col in ['time (months)', 'length_filtered']:
            rolling_mean, rolling_std, rolling_max, rolling_min = calculate_rolling_features(df, col,
                                                                                             self.min_sequence_length)
            static_mean, static_std, static_max, static_min = calculate_static_features(df, col)

            # Ajout des nouvelles caractéristiques
            df[f'rolling_mean_{col}'] = rolling_mean
            df[f'rolling_std_{col}'] = rolling_std
            df[f'rolling_max_{col}'] = rolling_max
            df[f'rolling_min_{col}'] = rolling_min

            df[f'static_mean_{col}'] = static_mean
            df[f'static_std_{col}'] = static_std
            df[f'static_max_{col}'] = static_max
            df[f'static_min_{col}'] = static_min

        # Remplacement des NaN
        rolling_columns = [f'rolling_mean_{col}', f'rolling_std_{col}', f'rolling_max_{col}', f'rolling_min_{col}',
                           f'static_mean_{col}', f'static_std_{col}', f'static_max_{col}', f'static_min_{col}']
        replace_nan_by_mean_and_one(df, rolling_columns)

        # Remplir NaN pour certaines colonnes spécifiques
        df[['time (months)', 'length_measured', 'length_filtered']] = df[
            ['time (months)', 'length_measured', 'length_filtered']].fillna(0)

        # Ajout de décalages et de ratios
        def add_shifts_and_ratios(df, columns, max_shift=6):
            for col in columns:
                # Ajout des colonnes décalées
                for shift in range(1, max_shift + 1):
                    shifted_col = df.groupby('item_id')[col].shift(shift)
                    df[f'{col}_shift_{shift}'] = shifted_col

                # Calcul des ratios entre les décalages
                for shift in range(1, max_shift):
                    df[f'{col}_ratio_{shift}-{shift + 1}'] = df[f'{col}_shift_{shift}'] / (
                                df[f'{col}_shift_{shift + 1}'] + 1e-9)
            df.fillna(0, inplace=True)
            return df

        shift_columns = ['length_filtered']
        df = add_shifts_and_ratios(df, shift_columns)

        # Décomposition de la série temporelle
        def decompose_time_series(df, time_col, value_col, period=12):
            # Vérifier la présence des colonnes
            if time_col not in df.columns or value_col not in df.columns:
                print(f"Les colonnes '{time_col}' ou '{value_col}' n'existent pas dans le DataFrame.")
                return df  # Retourner le DataFrame initial en cas d'erreur

            # Créer une copie du DataFrame pour les calculs
            df_copy = df.copy()

            # Définir la date de départ
            start_date = pd.Timestamp('2024-01-01')
            df_copy[time_col] = df_copy[time_col].apply(lambda x: start_date + pd.DateOffset(months=int(x)))

            # Trier et préparer le DataFrame
            df_copy = df_copy.sort_values(by=time_col)
            df_copy = df_copy.dropna(subset=[time_col, value_col])

            # Effectuer la décomposition STL
            try:
                stl = STL(df_copy[value_col], period=period)
                result = stl.fit()
            except Exception as e:
                print(f"Erreur lors de la décomposition STL: {e}")
                return df  # Retourner le DataFrame initial en cas d'erreur

            # Ajouter les résultats au DataFrame original
            df['Trend'] = result.trend
            df['Seasonal'] = result.seasonal
            df['Residual'] = result.resid

            return df

        df = decompose_time_series(df, 'time (months)', 'length_filtered')

        # Trier le DataFrame final
        df.sort_values(by=["item_id", "time (months)"], ascending=[True, True], inplace=True)

        return df