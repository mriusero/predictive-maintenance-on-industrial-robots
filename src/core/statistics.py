import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, chi2_contingency, pearsonr, wilcoxon, \
    friedmanchisquare
from statsmodels.stats.outliers_influence import variance_inflation_factor


class StatisticalTests:
    def __init__(self, df):
        """
        Initialize the StatisticalTests object with a DataFrame.
        :param df: The DataFrame containing the data to be analyzed.
        """
        self.df = df


    def test_multicollinearity(self):
        """
        Test for multicollinearity using the Variance Inflation Factor (VIF)
        for all numeric variables in the DataFrame.

        :return: A DataFrame containing the VIF values for each numeric variable.
        """

        numeric_df = self.df.select_dtypes(include=[np.number])              # Select only numeric columns
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()  # Handle infinite values

        vif_data = pd.DataFrame()                       # Compute VIF for each numeric variable
        vif_data["Variable"] = numeric_df.columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]

        st.write("Variance Inflation Factor (VIF):")        # Display the result with Streamlit
        return vif_data


    def test_normality(self, column_name):
        """
        Perform the Shapiro-Wilk test for normality on a specified column.

        :param column_name: The name of the column to test for normality.
        :return: A tuple (result, p_value), where `result` is True if normal, False otherwise,
                 and `p_value` is the p-value of the test.
        """
        stat, p_value = shapiro(self.df[column_name].dropna())  # Drop missing values
        alpha = 0.05                                            # Significance level
        return p_value > alpha, p_value


    def test_t_test(self, col1, col2):
        """
        Perform the Student's t-test for two independent samples.

        :param col1: The name of the first column to compare.
        :param col2: The name of the second column to compare.
        :return: A tuple (result, p_value), where `result` is True if not significant,
                 False if significant, and `p_value` is the p-value of the test.
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = ttest_ind(group1, group2)
        alpha = 0.05                                        # Significance level
        return p_value > alpha, p_value


    def test_mannwhitney(self, col1, col2):
        """
        Perform the Mann-Whitney U test for two independent samples.

        :param col1: The name of the first column to compare.
        :param col2: The name of the second column to compare.
        :return: A tuple (result, p_value), where `result` is True if not significant,
                 False if significant, and `p_value` is the p-value of the test.
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = mannwhitneyu(group1, group2)
        alpha = 0.05                                    # Significance level
        return p_value > alpha, p_value


    def test_anova(self, col):
        """
        Perform the ANOVA (Analysis of Variance) test to compare the means of multiple samples
        based on the specified column (e.g., 'Failure mode').

        :param col: The name of the column that defines the groups (categorical).
        :return: A tuple (result, p_value), where `result` is True if not significant,
                 False if significant, and `p_value` is the p-value of the test.

        :raises ValueError: If the specified column is not found in the DataFrame or is not categorical.
        """
        if col not in self.df.columns:                                                  # Check if the column exists
            raise ValueError(f"The column {col} does not exist in the DataFrame.")

        if not pd.api.types.is_categorical_dtype(self.df[col]) and not pd.api.types.is_object_dtype(self.df[col]):  # Check if the column is categorical or integer type
            if pd.api.types.is_integer_dtype(self.df[col]):
                self.df[col] = self.df[col].astype('category')
            else:
                raise ValueError(
                    f"The column {col} must be categorical or integer type for conversion.")

        numeric_cols = self.df.select_dtypes(include=['number']).columns        # Filter numeric columns

        if not numeric_cols.size:                                                           # Ensure there are numeric columns to perform ANOVA
            raise ValueError("No numeric columns are available for the ANOVA test.")

        groups = [self.df[self.df[col] == category][numeric_cols[0]] for category in self.df[col].unique()]     # Create groups for each category in the categorical column

        if any(len(group) < 2 for group in groups):         # Ensure each group has at least two values
            raise ValueError(
                "One or more groups have fewer than two values, which is insufficient for the ANOVA test.")

        stat, p_value = f_oneway(*groups)       # Perform the ANOVA test

        alpha = 0.05                            # Significance level
        return p_value > alpha, p_value


    def test_chi2(self, col1, col2):
        """
        Perform the Chi-square test for independence between two categorical variables.

        :param col1: The name of the first categorical column.
        :param col2: The name of the second categorical column.
        :return: A tuple (result, p_value), where `result` is True if not significant,
                 False if significant, and `p_value` is the p-value of the test.
        """
        contingency_table = pd.crosstab(self.df[col1], self.df[col2])           # Create a contingency table
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        alpha = 0.05                                                            # Significance level
        return p_value > alpha, p_value


    def test_correlation(self, col1, col2):
        """
        Perform the Pearson correlation test to measure the linear relationship between two continuous variables.

        :param col1: The name of the first continuous variable.
        :param col2: The name of the second continuous variable.
        :return: A tuple (corr_coef, p_value), where `corr_coef` is the correlation coefficient
                 and `p_value` is the p-value of the test.
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        corr_coef, p_value = pearsonr(group1, group2)
        return corr_coef, p_value


    def test_wilcoxon(self, col1, col2):
        """
        Perform the Wilcoxon signed-rank test to compare paired samples when the distribution is not normal.

        :param col1: The name of the first column of paired data.
        :param col2: The name of the second column of paired data.
        :return: A tuple (result, p_value), where `result` is True if not significant,
                 False if significant, and `p_value` is the p-value of the test.
        """
        group1 = self.df[col1]
        group2 = self.df[col2]
        stat, p_value = wilcoxon(group1, group2)
        alpha = 0.05  #                                 Significance level
        return p_value > alpha, p_value

    def test_friedman(self, subject_col='item_index', within_col='time (months)', dv_col='length_measured'):
        """
        Perform the Friedman test for comparing distributions of a dependent variable at different time points
        for each subject.

        :param subject_col: The column identifying the subjects (e.g., machines).
        :param within_col: The column representing the within-subject variable (e.g., time).
        :param dv_col: The dependent variable column (e.g., crack length).
        :return: A tuple (stat, p_value), where `stat` is the test statistic and `p_value` is the p-value.

        :raises ValueError: If any specified columns do not exist in the DataFrame.
        """
        for col in [subject_col, within_col, dv_col]:       # Check if the columns exist
            if col not in self.df.columns:
                raise ValueError(f"The column {col} does not exist in the DataFrame.")

        data_pivot = self.df.pivot(index=subject_col, columns=within_col, values=dv_col)                    # Reshape data for the Friedman test
        stat, p_value = friedmanchisquare(*[data_pivot[col].dropna() for col in data_pivot.columns])        # Perform the Friedman test

        return stat, p_value


def run_statistical_test(df, test_type, *args):
    """
    Run a specified statistical test on the DataFrame.

    :param df: The DataFrame containing the data to be analyzed.
    :param test_type: The type of statistical test to run (e.g., 'normality', 'ttest').
    :param args: Additional arguments needed for the specific test (e.g., column names).
    :return: The result of the test displayed using Streamlit.
    """
    tester = StatisticalTests(df)

    if test_type == 'normality':  # Shapiro-Wilk test for normality
        result, p_value = tester.test_normality(args[0])
        return st.write(f"--> Normality Test - `{args[0]}` - p-value: `{p_value}` - Result: {'`Normal`' if result else '`Not Normal`'}")

    elif test_type == 'ttest':  # T-test for independent samples
        result, p_value = tester.test_t_test(args[0], args[1])
        return st.write(f"T-Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'mannwhitney':  # Mann-Whitney U test for independent samples
        result, p_value = tester.test_mannwhitney(args[0], args[1])
        return st.write(f"Mann-Whitney Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'anova':  # ANOVA test for comparing multiple means
        result, p_value = tester.test_anova(args[0])
        return st.write(f"ANOVA Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'friedman':  # Friedman test for repeated measures
        stat, p_value = tester.test_friedman(*args)
        return st.write(f"Friedman Test - p-value: {p_value}, Statistic: {stat}")

    elif test_type == 'chi2':  # Chi-square test for categorical independence
        result, p_value = tester.test_chi2(args[0], args[1])
        return st.write(f"Chi-Square Test between {args[0]} and {args[1]} - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'correlation':  # Pearson correlation test for linear relationship
        corr_coef, p_value = tester.test_correlation(args[0], args[1])
        return st.write(f"Correlation Test between {args[0]} and {args[1]} - p-value: {p_value}, Correlation coefficient: {corr_coef}")

    elif test_type == 'wilcoxon':  # Wilcoxon signed-rank test for paired data
        result, p_value = tester.test_wilcoxon(args[0], args[1])
        return st.write(f"Wilcoxon Test - p-value: {p_value}, Result: {'Not Significant' if result else 'Significant'}")

    elif test_type == 'multicollinearity':  # Test for multicollinearity (VIF)
        vif_data = tester.test_multicollinearity()
        return st.dataframe(vif_data)
    else:
        return st.write("Unknown test type")