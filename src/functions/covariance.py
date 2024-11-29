import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_covariance_matrix(df, exclude_columns=None):
    """
    Calcule et visualise la matrice de covariance des variables numériques du DataFrame,
    en excluant les colonnes spécifiées.

    :param df: DataFrame contenant les données
    :param exclude_columns: Liste des noms de colonnes à exclure du calcul de la covariance.
                            Par défaut, aucune colonne n'est exclue.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Assurer que les colonnes à exclure sont dans le DataFrame
    exclude_columns = [col for col in exclude_columns if col in df.columns]

    # Sélectionner toutes les colonnes numériques sauf celles à exclure
    numeric_columns = df.select_dtypes(include=[float, int]).columns.difference(exclude_columns)

    if numeric_columns.empty:
        raise ValueError("Aucune colonne numérique restante après exclusion. Veuillez vérifier vos données.")

    # Calculer la matrice de covariance
    covariance_matrix = df[numeric_columns].cov()

    # Création de la figure et de l'axe
    fig = plt.figure(figsize=(10, 8))

    # Tracer la heatmap
    sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0)

    # Ajouter des labels et un titre
    plt.title('Matrice de Covariance')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    # Afficher le graphique
    st.pyplot(fig)

# Exemple d'utilisation
# df = pd.DataFrame({
#     'var1': [1, 2, 3, 4, 5],
#     'var2': [5, 4, 3, 2, 1],
#     'var3': [2, 3, 4, 5, 6],
#     'text_col': ['A', 'B', 'C', 'D', 'E']  # Colonne à exclure
# })

# plot_covariance_matrix(df, exclude_columns=['text_col'])
