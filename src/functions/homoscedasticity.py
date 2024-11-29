import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def check_homoscedasticity(df, target_col, exclude_columns=None):
    """
    Vérifie l'homoscedasticité des résidus en traçant un graphique de résidus versus valeurs ajustées,
    après nettoyage des données pour les valeurs infinies et manquantes.

    :param df: DataFrame contenant les données
    :param target_col: Nom de la colonne cible (variable dépendante)
    :param exclude_columns: Liste des noms de colonnes à exclure du modèle de régression.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Assurer que les colonnes à exclure sont dans le DataFrame
    exclude_columns = [col for col in exclude_columns if col in df.columns]

    # Sélectionner toutes les colonnes numériques sauf celles à exclure
    numeric_columns = df.select_dtypes(include=[float, int]).columns.difference(exclude_columns)

    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans le DataFrame.")

    # Nettoyer les données : supprimer les lignes avec NaN ou inf
    df_clean = df.dropna(subset=[target_col] + list(numeric_columns))
    df_clean = df_clean[~df_clean.isin([float('inf'), -float('inf')]).any(axis=1)]

    # Construire la matrice de variables indépendantes X
    X = df_clean[numeric_columns]
    X = sm.add_constant(X)  # Ajouter une constante (intercept) au modèle

    # Construire la variable dépendante y
    y = df_clean[target_col]

    # Ajuster le modèle de régression
    model = sm.OLS(y, X).fit()

    # Obtenir les valeurs ajustées et les résidus
    fitted_values = model.fittedvalues
    residuals = model.resid

    # Créer la figure et les axes
    fig = plt.figure(figsize=(12, 6))

    # Tracer les résidus versus valeurs ajustées
    sns.scatterplot(x=fitted_values, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')

    # Ajouter des labels et un titre
    plt.title('Résidus versus Valeurs Ajustées')
    plt.xlabel('Valeurs Ajustées')
    plt.ylabel('Résidus')

    # Afficher le graphique
    st.pyplot(fig)

# Exemple d'utilisation
# df = pd.DataFrame({
#     'var1': [1, 2, 3, 4, 5],
#     'var2': [5, 4, 3, 2, 1],
#     'var3': [2, 3, 4, 5, 6],
#     'target': [2.3, 3.1, 3.6, 4.5, 5.0]  # Variable cible
# })

# check_homoscedasticity(df, target_col='target', exclude_columns=['var3'])
