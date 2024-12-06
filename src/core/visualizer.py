import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
import seaborn as sns
import pandas as pd
from src.forecasting import clean_data, FeatureAdder
from ..core import dataframing_data


class DataVisualizer:
    """
    A class that provides data visualization methods for plotting dataframes.

    Attributes:
        dataframes (dict): A dictionary containing various dataframes.
        feature_adder (FeatureAdder): An instance of FeatureAdder for adding features.
        df (dict): A dictionary containing preprocessed dataframes.
    """

    def __init__(self):
        """
        Initializes the DataVisualizer with predefined data and preprocessing.

        Sets up the dataframes for training, testing, and pseudo-testing with added features.
        """
        self.dataframes = dataframing_data()
        self.feature_adder = FeatureAdder(min_sequence_length=2)
        self.df = {
            'train': self.preprocessing(self.dataframes['train']),
            'pseudo_test': self.preprocessing(self.dataframes['pseudo_test']),
            'pseudo_test_with_truth': self.preprocessing(self.dataframes['pseudo_test_with_truth']),
            'test': self.preprocessing(self.dataframes['test'])
        }

    @st.cache_data
    def preprocessing(_self, df):
        """
        Preprocesses a given dataframe by renaming columns, sorting, and adding features.
        :param df: The dataframe to preprocess.
        :return: The preprocessed dataframe with added features.
        """
        df = df.rename(columns={'crack length (arbitary unit)': 'length_measured'})
        df = df.sort_values(by=['item_id', 'time (months)'])
        df['item_id'] = df['item_id'].astype(int)
        return _self.feature_adder.add_features(clean_data(df), particles_filtery=True)


    def get_dataframes(self):
        """
        Returns the original dataframes.
        :return: The dictionary of original dataframes.
        """
        return self.dataframes


    def get_the(self, df_key):
        """
        Returns a specific preprocessed dataframe by its key.
        :param df_key: The key of the dataframe to retrieve.
        :return: The preprocessed dataframe corresponding to the key.
        """
        return self.df[df_key]


    def get_color_palette(self):
        """
        The function provides suitable palettes for categorical, continuous, cyclical, and sequential data types,
        with considerations for color blindness accessibility.

        :return: A dictionary containing various color palettes.
        """
        return {
            'qualitative': {  # Categorical data
                'D3': px.colors.qualitative.D3,  # Color blindness friendly
                'T10': px.colors.qualitative.T10,  # Color blindness friendly
                'Safe': px.colors.qualitative.Safe  # Color blindness friendly
            },
            'continuous': {  # Continuous data
                'Viridis': px.colors.sequential.Viridis,  # Color blindness friendly
                'Cividis': px.colors.sequential.Cividis,  # Color blindness friendly
                'Inferno': px.colors.sequential.Inferno,  # Color blindness friendly
                'Magma': px.colors.sequential.Magma,  # Color blindness friendly
                'Plasma': px.colors.sequential.Plasma,  # Color blindness friendly
                'Turbo': px.colors.sequential.Turbo  # Color blindness friendly
            },
            'cyclical': {  # Cyclical data
                'IceFire': px.colors.cyclical.IceFire,  # Color blindness friendly (best option)
            },
            'sequential': {  # Sequential data
                'Viridis': px.colors.sequential.Viridis,  # Color blindness friendly
                'Cividis': px.colors.sequential.Cividis,  # Color blindness friendly
                'Inferno': px.colors.sequential.Inferno,  # Color blindness friendly
                'Magma': px.colors.sequential.Magma,  # Color blindness friendly
                'Plasma': px.colors.sequential.Plasma,  # Color blindness friendly
                'Blues': px.colors.sequential.Blues,  # Color blindness friendly
                'Greys': px.colors.sequential.Greys  # Color blindness friendly
            }
        }

    ## --- 1) SCATTER PLOT ---
    def plot_scatter_with_color(self, df_key, x_col, y_col, color_col, palette_type, palette_name):
        """
        Plots a scatter plot with customized coloring.

        :param df_key: Key for the dataframe to plot.
        :param x_col: The column name for the x-axis.
        :param y_col: The column name for the y-axis.
        :param color_col: The column used for coloring the points.
        :param palette_type: Type of the color palette ('continuous' or 'categorical').
        :param palette_name: The specific palette name to use.
        """
        df = self.df[df_key]
        color_palette = self.get_color_palette()

        if x_col and y_col and color_col:
            if palette_type == 'continuous':
                color_scale = color_palette['continuous'].get(palette_name, px.colors.cyclical.mrybm_r)
                color_discrete_map = None  # No discrete colors for continuous palette
            else:
                color_scale = None  # No continuous color for a discrete palette
                color_discrete_map = {
                    val: color_palette['qualitative'][palette_name][i % len(color_palette['qualitative'][palette_name])]
                    for i, val in enumerate(df[color_col].unique())
                }
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                title=f'Scatter plot for {x_col} vs {y_col}',
                color_continuous_scale=color_scale,
                color_discrete_map=color_discrete_map
            )
            st.plotly_chart(fig)


    ## --- 2) HISTOGRAM ---
    def plot_histogram_with_color(self, df_key, x_col, y_col, color_col, palette_type, palette_name):
        """
        Plots a histogram with customized coloring.

        :param df_key: Key for the dataframe to plot.
        :param x_col: The column name for the x-axis.
        :param y_col: The column name for the y-axis.
        :param color_col: The column used for coloring the bars.
        :param palette_type: Type of the color palette ('qualitative', 'continuous', 'cyclical', or 'sequential').
        :param palette_name: The specific palette name to use.
        :raises ValueError: If an invalid palette type or name is provided.
        """
        df = self.df[df_key]
        color_palette = self.get_color_palette()

        if palette_type not in color_palette:
            raise ValueError(f"Invalid palette type. Choose from {list(color_palette.keys())}.")

        if palette_name not in color_palette[palette_type]:
            raise ValueError(
                f"Palette '{palette_name}' not found for type '{palette_type}'. Choose from {list(color_palette[palette_type].keys())}.")

        palette = color_palette[palette_type][palette_name]
        color_map = None

        if palette_type == 'qualitative':
            # For qualitative palettes, `color_discrete_map` must map values to colors
            color_map = {val: palette[i % len(palette)] for i, val in enumerate(df[color_col].unique())}

        fig = px.histogram(
            df, x=x_col, y=y_col, color=color_col,
            color_discrete_map=color_map,
            title=f'Distribution of {y_col} per {x_col}',
            marginal="box",
            hover_data=df.columns
        )
        st.plotly_chart(fig)


    ## --- 3) PAIRPLOT ---
    def plot_pairplot(self, data, hue=None, palette='Set2'):
        """
        Plots a pair plot from a given Pandas DataFrame.

        :param data: DataFrame containing the data.
        :param hue: Name of the column for coloring the points based on a categorical variable.
        :param palette: The color palette to use for the plot.
        """
        sns.set_theme(style='darkgrid', rc={
            'axes.facecolor': '#313234',  # Background color of the axes
            'figure.facecolor': '#313234',  # Background color of the figure
            'axes.labelcolor': 'white',  # Color of the axis labels
            'xtick.color': 'white',  # Color of x-axis ticks
            'ytick.color': 'white',  # Color of y-axis ticks
            'grid.color': '#444444',  # Grid line color
            'text.color': 'white'  # Text color
        })
        fig = sns.pairplot(data, hue=hue, palette=palette)
        st.pyplot(fig)


        ## --- SPECIFIC ---
    def plot_correlation_matrix(self, df_key):
        """
        Plots a correlation matrix for numerical variables in the dataframe.
        :param df_key: The key of the dataframe to use.
        """
        df = self.df[df_key]
        color_palette = self.get_color_palette()
        numeric_df = df.select_dtypes(include=[float, int])
        corr_matrix = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation'),
            text=corr_matrix.round(2).astype(str).values,
            texttemplate='%{text}',
            textfont=dict(size=12, color='white'),
            hoverinfo='text',
            showscale=True
        ))
        fig.update_layout(
            title='Correlation Matrix',
            xaxis_title='Variables',
            yaxis_title='Variables',
            xaxis=dict(ticks='', side='bottom', title_standoff=0),
            yaxis=dict(ticks='', side='left', title_standoff=0),
            plot_bgcolor='#313234',
            paper_bgcolor='#313234',
            font=dict(color='white'),
            title_font=dict(color='white'),
            coloraxis_colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1', '0', '1'])
        )
        st.plotly_chart(fig, use_container_width=True)


    def plot_correlation_with_target(self, df_key, target_variable):
        """
        Plots correlation coefficients between the target variable and other numerical variables.

        :param df_key: The key of the dataframe to use.
        :param target_variable: The target variable to correlate with.
        """
        df = self.df[df_key]

        numeric_df = df.select_dtypes(include=[float, int])

        if target_variable not in numeric_df.columns:
            st.error(f"The target variable '{target_variable}' is not present in the DataFrame.")
            return

        correlations = numeric_df.corr()[target_variable].dropna().sort_values(ascending=False)
        correlations = correlations[correlations.index != target_variable]

        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"### Correlations with '{target_variable}'")
            st.dataframe(correlations)
        with col2:
            fig = go.Figure(data=go.Bar(
                x=correlations.index,
                y=correlations.values,
                marker_color='#00d2ba'
            ))
            fig.update_layout(
                title=f'Correlation Coefficients with {target_variable}',
                xaxis_title='Variables',
                yaxis_title='Correlation Coefficient',
                plot_bgcolor='#313234',
                paper_bgcolor='#313234',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)


    def boxplot(self, df_key, x_col, y_col):
        """
        Plots a boxplot of x_col against y_col.

        :param df_key: The key of the dataframe to use.
        :param x_col: The column of the dataframe to use for the x-axis.
        :param y_col: The column of the dataframe to use for the y-axis.
        """
        data = self.df[df_key]
        fig = plt.figure(figsize=(10, 6))  # Set figure size

        sns.boxplot(x=x_col, y=y_col, data=data, hue=x_col, palette="Set2",
                    legend=False)  # Create boxplot using seaborn

        plt.xlabel(x_col)  # Label for x-axis
        plt.ylabel(y_col)  # Label for y-axis
        plt.title(f'Boxplot of {x_col} vs {y_col}')  # Plot title

        st.pyplot(fig)  # Display the plot


    def decompose_time_series(self, df_key, time_col, value_col, period=12):
        """
        Decomposes a time series into trend, seasonal, and residual components.

        :param df_key: The key of the dataframe to use.
        :param time_col: The column containing time values.
        :param value_col: The column containing the values to decompose.
        :param period: The period for the decomposition (default is 12).
        """
        df = self.df[df_key]

        if time_col not in df.columns or value_col not in df.columns:
            st.error(f"The columns '{time_col}' or '{value_col}' do not exist in the DataFrame.")
            return

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(by=time_col)
        df = df.dropna(subset=[time_col, value_col])
        df.set_index(time_col, inplace=True)

        # Perform STL decomposition
        try:
            stl = STL(df[value_col], period=period)
            result = stl.fit()
        except Exception as e:
            st.error(f"Error during STL decomposition: {e}")
            return

        # Plot the results
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f'Time Series Decomposition: {value_col}', fontsize=16)

        # Trend
        axes[0].plot(result.trend, label='Trend', color='blue')
        axes[0].set_title('Trend')
        axes[0].legend(loc='upper left')

        # Seasonal
        axes[1].plot(result.seasonal, label='Seasonality', color='green')
        axes[1].set_title('Seasonality')
        axes[1].legend(loc='upper left')

        # Residual
        axes[2].plot(result.resid, label='Residual', color='red')
        axes[2].set_title('Residual')
        axes[2].legend(loc='upper left')

        # Original series
        axes[3].plot(df[value_col], label='Original Series', color='gray')
        axes[3].set_title('Original Series')
        axes[3].legend(loc='upper left')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig)