import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import streamlit as st


class DisplayData:
    def __init__(self, data):
        self.data = data

    def plot_discrete_scatter(self, df, x_col, y_col, color_col):

        if x_col and y_col and color_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f'Nuage de points pour {x_col} vs {y_col}')
            st.plotly_chart(fig)

    def plot_correlation_matrix(self):
        numeric_df = self.data.select_dtypes(include=[float, int])
        corr = numeric_df.corr()
        fig = px.imshow(corr, color_continuous_scale='Viridis', text_auto=True)
        fig.update_layout(title='Correlation Matrix', title_x=0.5)
        return st.plotly_chart(fig)

    def plot_learning_curve(self, model, X, y, cv=5):
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
        fig = px.histogram(residuals, nbins=30, color_discrete_sequence=['green'])
        fig.update_layout(title='Histogram of Residuals', xaxis_title='Residuals', yaxis_title='Frequency')
        return st.plotly_chart(fig)

    def plot_density_curve_of_residuals(self, residuals):
        fig = px.density_contour(x=residuals, color_discrete_sequence=['green'])
        fig.update_layout(title='Density Curve of Residuals', xaxis_title='Residuals', yaxis_title='Density')
        return st.plotly_chart(fig)

    def plot_qq_diagram(self, residuals):
        fig = go.Figure()

        # QQ plot
        qq = stats.probplot(residuals, dist="norm", plot=None)
        x = qq[0][0]  # Quantiles théoriques
        y = qq[0][1]  # Quantiles observés

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='QQ Plot'))
        fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[min(x), max(x)], mode='lines', line=dict(color='red'),
                                 name='Line of Equality'))

        fig.update_layout(title='QQ Plot des résidus', xaxis_title='Quantiles Théoriques',
                          yaxis_title='Quantiles Observés')

        return st.plotly_chart(fig)

    def plot_predictions_histograms(self, true_rul, predicted_rul):
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
        df = self.data

        if column_name not in df.columns:
            raise ValueError(f"La colonne '{column_name}' n'existe pas dans le DataFrame.")

        data = df[column_name]

        histogram = go.Histogram(
            x=data,
            histnorm='probability density',
            name='Histogramme',
            opacity=0.75
        )
        kde = gaussian_kde(data, bw_method='scott')
        x_values = np.linspace(min(data), max(data), 1000)
        kde_values = kde(x_values)
        curve = go.Scatter(
            x=x_values,
            y=kde_values,
            mode='lines',
            name='Courbe de distribution',
            line=dict(color='red')
        )
        fig = go.Figure(data=[histogram, curve])
        fig.update_layout(
            title=f'Distribution de la colonne {column_name}',
            xaxis_title=column_name,
            yaxis_title='Densité',
            template='plotly_white'
        )
        return st.plotly_chart(fig)

