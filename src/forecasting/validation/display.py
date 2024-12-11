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

