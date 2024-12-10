import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from scipy.stats import gaussian_kde  # Ensure this import for KDE in `plot_distribution_histogram`
from sklearn.model_selection import learning_curve


class DisplayData:
    def __init__(self, data):
        """
        Initializes DisplayData class with a pandas DataFrame.

        :param data: The pandas DataFrame containing the data to visualize.
        """
        self.data = data

    def plot_discrete_scatter(self, df, x_col, y_col, color_col):
        """
        Plots a scatter plot of discrete data.

        :param df: The DataFrame containing the data.
        :param x_col: Column name for the x-axis.
        :param y_col: Column name for the y-axis.
        :param color_col: Column name for the color grouping.
        """
        if x_col and y_col and color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f'Scatter plot for {x_col} vs {y_col}')
            st.plotly_chart(fig)

    def plot_correlation_matrix(self):
        """
        Plots a correlation matrix heatmap for numeric columns in the DataFrame.
        """
        numeric_df = self.data.select_dtypes(include=[float, int])
        corr = numeric_df.corr()
        fig = px.imshow(corr, color_continuous_scale='Viridis', text_auto=True)
        fig.update_layout(title='Correlation Matrix', title_x=0.5)
        return st.plotly_chart(fig)

    def plot_learning_curve(self, model, X, y, cv=5):
        """
        Plots the learning curve for a given model.

        :param model: The machine learning model.
        :param X: Features for training the model.
        :param y: Target variable.
        :param cv: Number of cross-validation folds.
        """
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training score',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean, mode='lines+markers', name='Cross-validation score',
                                 line=dict(color='blue')))
        fig.update_layout(title='Learning Curve', xaxis_title='Training Size', yaxis_title='Score')
        return st.plotly_chart(fig)

    def plot_scatter_real_vs_predicted(self, y_test, predictions):
        """
        Plots a scatter plot comparing real vs predicted values.

        :param y_test: Actual target values.
        :param predictions: Predicted values.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=y_test, y=predictions, mode='markers', marker=dict(color='cyan'), name='Predicted vs Actual'))
        fig.add_trace(
            go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], mode='lines', line=dict(color='red'),
                       name='Ideal Line'))
        fig.update_layout(title='Scatter Plot of Real vs Predicted', xaxis_title='Real Values',
                          yaxis_title='Predicted Values')
        return st.plotly_chart(fig)

    def plot_residuals_vs_predicted(self, y_test, predictions):
        """
        Plots residuals vs predicted values.

        :param y_test: Actual target values.
        :param predictions: Predicted values.
        """
        residuals = y_test - predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predictions, y=residuals, mode='markers', marker=dict(color='cyan'),
                                 name='Residuals vs Predicted'))
        fig.add_trace(go.Scatter(x=[min(predictions), max(predictions)], y=[0, 0], mode='lines', line=dict(color='red'),
                                 name='Zero Line'))
        fig.update_layout(title='Residuals vs Predicted Values', xaxis_title='Predicted Values',
                          yaxis_title='Residuals')
        return st.plotly_chart(fig)

    def plot_histogram_of_residuals(self, residuals):
        """
        Plots a histogram of residuals.

        :param residuals: Array of residuals.
        """
        fig = px.histogram(residuals, nbins=30, color_discrete_sequence=['green'])
        fig.update_layout(title='Histogram of Residuals', xaxis_title='Residuals', yaxis_title='Frequency')
        return st.plotly_chart(fig)

    def plot_density_curve_of_residuals(self, residuals):
        """
        Plots a density curve for the residuals.

        :param residuals: Array of residuals.
        """
        fig = px.density_contour(x=residuals, color_discrete_sequence=['green'])
        fig.update_layout(title='Density Curve of Residuals', xaxis_title='Residuals', yaxis_title='Density')
        return st.plotly_chart(fig)

    def plot_qq_diagram(self, residuals):
        """
        Plots a QQ plot for the residuals to assess normality.

        :param residuals: Array of residuals.
        """
        fig = go.Figure()

        # QQ plot
        qq = stats.probplot(residuals, dist="norm", plot=None)
        x = qq[0][0]  # Theoretical quantiles
        y = qq[0][1]  # Observed quantiles

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='QQ Plot'))
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[min(x), max(x)], mode='lines', line=dict(color='red'),
                                 name='Line of Equality'))

        fig.update_layout(title='QQ Plot of Residuals', xaxis_title='Theoretical Quantiles',
                          yaxis_title='Observed Quantiles')

        return st.plotly_chart(fig)

    def plot_predictions_histograms(self, true_rul, predicted_rul):
        """
        Plots histograms comparing true vs predicted RUL (Remaining Useful Life).

        :param true_rul: True RUL values.
        :param predicted_rul: Predicted RUL values.
        """
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=true_rul, nbinsx=30, name='True RUL', opacity=0.5, marker_color='blue'))
        fig.add_trace(
            go.Histogram(x=predicted_rul, nbinsx=30, name='Predicted RUL', opacity=0.5, marker_color='#ff322a'))
        fig.update_layout(
            title='Distribution of True and Predicted RUL',
            xaxis_title='RUL (months)',
            yaxis_title='Frequency',
            barmode='overlay'
        )
        return st.plotly_chart(fig)

    def plot_distribution_histogram(self, column_name):
        """
        Plots a histogram of a specified column's distribution with a KDE curve.

        :param column_name: Name of the column to plot.
        """
        df = self.data

        if column_name not in df.columns:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")

        data = df[column_name]

        histogram = go.Histogram(
            x=data,
            histnorm='probability density',
            name='Histogram',
            opacity=0.75
        )
        kde = gaussian_kde(data, bw_method='scott')
        x_values = np.linspace(min(data), max(data), 1000)
        kde_values = kde(x_values)
        curve = go.Scatter(
            x=x_values,
            y=kde_values,
            mode='lines',
            name='Density Curve',
            line=dict(color='red')
        )
        fig = go.Figure(data=[histogram, curve])
        fig.update_layout(
            title=f'Distribution of {column_name}',
            xaxis_title=column_name,
            yaxis_title='Density',
            template='plotly_white'
        )
        return st.plotly_chart(fig)

    def plot_scatter_with_color(self, df, x_col, y_col, color_col, palette_type, palette_name='Safe'):
        """
        Plots a scatter plot with customized coloring.

        :param df: DataFrame to plot.
        :param x_col: The column name for the x-axis.
        :param y_col: The column name for the y-axis.
        :param color_col: The column used for coloring the points.
        :param palette_type: Type of the color palette ('continuous' or 'categorical').
        :param palette_name: The specific palette name to use (default: 'Safe' for categorical).
        """

        if x_col and y_col and color_col:
            if palette_type == 'continuous':
                # Use a default continuous color palette if specific palette is not provided
                color_scale = px.colors.cyclical.mrybm_r
                color_discrete_map = None  # No discrete colors for continuous palette
            else:
                # Use a default qualitative palette 'Safe'
                default_palette = px.colors.qualitative.Safe
                color_discrete_map = {
                    val: default_palette[i % len(default_palette)]
                    for i, val in enumerate(df[color_col].unique())
                }
                color_scale = None  # No continuous color for a discrete palette

            # Generate the scatter plot
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                title=f'Scatter plot for {x_col} vs {y_col}',
                color_continuous_scale=color_scale,
                color_discrete_map=color_discrete_map
            )
            st.plotly_chart(fig)

    def plot_histogram_with_color(self, df, x_col, y_col, color_col, palette_type, palette_name='Safe'):
        """
        Plots a histogram with customized coloring.
        :param df: Key for the dataframe to plot.
        :param x_col: The column name for the x-axis.
        :param y_col: The column name for the y-axis.
        :param color_col: The column used for coloring the bars.
        :param palette_type: Type of the color palette ('qualitative', 'continuous', 'cyclical', or 'sequential').
        :param palette_name: The specific palette name to use (default: 'Safe' for qualitative).
        :raises ValueError: If an invalid palette type is provided.
        """

        # Define default palettes directly
        qualitative_palette = px.colors.qualitative.Safe
        continuous_palette = px.colors.cyclical.mrybm_r
        sequential_palette = px.colors.sequential.Viridis
        cyclical_palette = px.colors.cyclical.mygbm
        # Choose the palette based on the type
        if palette_type == 'qualitative':
            palette = qualitative_palette
            color_map = {
                val: palette[i % len(palette)]
                for i, val in enumerate(df[color_col].unique())
            }
            color_discrete_map = color_map
        elif palette_type == 'continuous':
            color_discrete_map = None
            palette = continuous_palette
        elif palette_type == 'cyclical':
            color_discrete_map = None
            palette = cyclical_palette
        elif palette_type == 'sequential':
            color_discrete_map = None
            palette = sequential_palette
        else:
            raise ValueError(
                "Invalid palette type. Choose 'qualitative', 'continuous', 'cyclical', or 'sequential'.")
        # Generate the histogram
        fig = px.histogram(
            df, x=x_col, y=y_col, color=color_col,
            color_discrete_map=color_discrete_map,
            title=f'Distribution of {y_col} per {x_col}',
            marginal="box",
            hover_data=df.columns
        )
        st.plotly_chart(fig)

