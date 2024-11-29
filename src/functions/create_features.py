import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def create_features(df, time_col, value_col, item_col, period=12):
    # Convertir la colonne de temps en datetime si ce n'est pas déjà fait
    #1df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(by=[item_col, time_col])

    # Créer des colonnes pour les composants de la décomposition STL
    df.set_index(time_col, inplace=True)

    # Décomposition STL par item
    items = df[item_col].unique()
    for item in items:
        item_df = df[df[item_col] == item]

        try:
            stl = STL(item_df[value_col], period=period)
            result = stl.fit()

            df.loc[item_df.index, 'trend'] = result.trend
            df.loc[item_df.index, 'seasonal'] = result.seasonal
            df.loc[item_df.index, 'residual'] = result.resid
            df.loc[item_df.index, 'deviation_from_trend'] = item_df[value_col] - result.trend

        except Exception as e:
            print(f"Erreur pour l'item {item}: {e}")

    # Calcul des statistiques descriptives
    df['rolling_mean'] = df[value_col].rolling(window=3).mean()
    df['rolling_std'] = df[value_col].rolling(window=3).std()
    df['rolling_median'] = df[value_col].rolling(window=3).median()

    # Normalisation et standardisation
    scaler_min_max = MinMaxScaler()
    scaler_standard = StandardScaler()

    numeric_columns = [value_col, 'trend', 'seasonal', 'residual']
    df[numeric_columns] = scaler_min_max.fit_transform(df[numeric_columns])
    df[numeric_columns] = scaler_standard.fit_transform(df[numeric_columns])

    return df

# Exemple d'utilisation
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=24, freq='M'),
#     'crack length (arbitary unit)': np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(size=24),
#     'item_index': np.random.randint(1, 4, size=24)  # Exemple d'index d'item
# })

# df_with_features = create_features(df, 'date', 'crack length (arbitary unit)', 'item_index')
# st.dataframe(df_with_features)
