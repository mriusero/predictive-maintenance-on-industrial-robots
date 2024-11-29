import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, chi2_contingency, pearsonr, wilcoxon, friedmanchisquare
from statsmodels.stats.anova import AnovaRM
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
class StatisticalTests:
    def __init__(self, df):
        self.df = df

    def test_multicollinearity(self):
        """
        Test de multicolinéarité utilisant le Variance Inflation Factor (VIF)
        pour toutes les variables numériques du DataFrame.
        """
        # Sélectionner uniquement les colonnes numériques
        numeric_df = self.df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()
        # Calculer le VIF pour chaque variable numérique
        vif_data = pd.DataFrame()
        vif_data["Variable"] = numeric_df.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]

        # Afficher le résultat avec Streamlit
        st.write("Variance Inflation Factor (VIF):")
        return vif_data

    def test_normality(self, column_name):
        """
        Test de normalité de Shapiro-Wilk
        """
        stat, p_value = shapiro(self.df[column_name].dropna())
        alpha = 0.05
        return p_value > alpha, p_value

    def test_t_test(self, col1, col2):
        """
        Test t de Student pour deux échantillons indépendants
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = ttest_ind(group1, group2)
        alpha = 0.05
        return p_value > alpha, p_value

    def test_mannwhitney(self, col1, col2):
        """
        Test de Mann-Whitney pour deux échantillons indépendants
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = mannwhitneyu(group1, group2)
        alpha = 0.05
        return p_value > alpha, p_value

    def test_anova(self, col):
        """
        Test ANOVA (Analyse de la Variance) pour comparer les moyennes de plusieurs échantillons
        en fonction de la colonne spécifiée (par exemple, 'Failure mode').
        """
        # Vérifier que la colonne existe dans le DataFrame
        if col not in self.df.columns:
            raise ValueError(f"La colonne {col} n'existe pas dans le DataFrame.")

        # Vérifier que la colonne est bien catégorique
        if not pd.api.types.is_categorical_dtype(self.df[col]) and not pd.api.types.is_object_dtype(self.df[col]):
            if pd.api.types.is_integer_dtype(self.df[col]):
                self.df[col] = self.df[col].astype('category')
            else:
                raise ValueError(
                    f"La colonne {col} doit être de type catégorique ou une colonne d'entiers pour la conversion.")

        # Filtrer le DataFrame pour ne conserver que les colonnes numériques
        numeric_cols = self.df.select_dtypes(include=['number']).columns

        # Assurer que la colonne à tester est numérique
        if not numeric_cols.size:
            raise ValueError("Aucune colonne numérique n'est disponible pour le test ANOVA.")

        # Créer une liste de séries pour chaque groupe en utilisant les colonnes numériques
        groups = [self.df[self.df[col] == category][numeric_cols[0]] for category in self.df[col].unique()]

        # Assurer que chaque groupe a au moins deux valeurs pour le test ANOVA
        if any(len(group) < 2 for group in groups):
            raise ValueError(
                "Un ou plusieurs groupes ont moins de deux valeurs, ce qui est insuffisant pour le test ANOVA.")

        # Effectuer le test ANOVA
        stat, p_value = f_oneway(*groups)

        # Seuil de significativité
        alpha = 0.05

        # Retourner le résultat
        return p_value > alpha, p_value

    def test_chi2(self, col1, col2):
        """
        Test du Chi-carré pour tester l'indépendance entre deux variables catégorielles
        """
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        alpha = 0.05
        return p_value > alpha, p_value

    def test_correlation(self, col1, col2):
        """
        Test de corrélation de Pearson pour mesurer la relation linéaire entre deux variables quantitatives
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        corr_coef, p_value = pearsonr(group1, group2)
        return corr_coef, p_value

    def test_wilcoxon(self, col1, col2):
        """
        Test de Wilcoxon pour comparer deux échantillons appariés
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = wilcoxon(group1, group2)
        alpha = 0.05
        return p_value > alpha, p_value

    def test_friedman(self, subject_col='item_index', within_col='time (months)', dv_col='length_measured'):
        """
        Test de Friedman pour comparer les distributions de 'Crack_Length' sur différents points de mesure ('Time') pour chaque sujet ('Machine').

        Arguments:
        - subject_col: Colonne identifiant les sujets (par ex. les machines).
        - within_col: Colonne représentant la variable intra-sujet (par ex. le temps).
        - dv_col: Colonne de la variable dépendante (par ex. la longueur de fissure).
        """
        # Vérifier que les colonnes existent dans le DataFrame
        for col in [subject_col, within_col, dv_col]:
            if col not in self.df.columns:
                raise ValueError(f"La colonne {col} n'existe pas dans le DataFrame.")

        # Réorganiser les données pour le test de Friedman
        data_pivot = self.df.pivot(index=subject_col, columns=within_col, values=dv_col)

        # Effectuer le test de Friedman
        stat, p_value = friedmanchisquare(*[data_pivot[col].dropna() for col in data_pivot.columns])

        return stat, p_value


def run_statistical_test(df, test_type, *args):
    tester = StatisticalTests(df)

    if test_type == 'normality':         # Test de normalité pour vérifier si la distribution des données suit une loi normale.
        result, p_value = tester.test_normality(args[0])
        return st.write(f"--> Normality Test - `{args[0]}` - p-value: `{p_value}` - Result: {'`Normal`' if result else '`Not Normal`'}")

    elif test_type == 'ttest':          # Test T (Student) pour comparer les moyennes de deux échantillons indépendants ou appariés.
        result, p_value = tester.test_t_test(args[0], args[1])
        return st.write(f"T-Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'mannwhitney':        # Test de Mann-Whitney pour comparer les distributions de deux échantillons indépendants.
        result, p_value = tester.test_mannwhitney(args[0], args[1])
        return st.write(f"Mann-Whitney Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'anova':              # Test ANOVA (Analyse de la Variance) pour comparer les moyennes de trois échantillons ou plus.
        result, p_value = tester.test_anova(args[0])
        return st.write(f"ANOVA Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'friedman':
        stat, p_value = tester.test_friedman(*args)
        return st.write(f"Friedman Test - p-value: {p_value}, Statistic: {stat}")

    elif test_type == 'chi2':               # Test du Chi-carré pour tester l'indépendance entre deux variables catégorielles.
        result, p_value = tester.test_chi2(args[0], args[1])
        return st.write(f"Chi-Square Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'correlation':         # Test de corrélation (Pearson) pour mesurer la force et la direction de la relation linéaire entre deux variables.
        corr_coef, p_value = tester.test_correlation(args[0], args[1])
        return st.write(f"Correlation Test between {args[0]} and {args[1]} - p-value: {p_value}, Correlation coefficient: {corr_coef}")

    elif test_type == 'wilcoxon':           # Test de Wilcoxon pour comparer les paires de valeurs lorsque la distribution n'est pas normale.
        result, p_value = tester.test_wilcoxon(args[0], args[1])
        return st.write(f"Wilcoxon Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'multicollinearity':  # Test de multicolinéarité (VIF)
        vif_data = tester.test_multicollinearity()
        return st.dataframe(vif_data)
    else:
        return st.write("Unknown test type")
