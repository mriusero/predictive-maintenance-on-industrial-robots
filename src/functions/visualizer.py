import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL
import seaborn as sns
import pandas as pd
from src.FailuresDetectModel import clean_data, FeatureAdder
from ..functions import dataframing_data


class DataVisualizer:
    def __init__(self):
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
        df = df.rename(columns={'crack length (arbitary unit)': 'length_measured'})
        df = df.sort_values(by=['item_id', 'time (months)'])
        df['item_id'] = df['item_id'].astype(int)
        return _self.feature_adder.add_features(clean_data(df), particles_filtery=True)

    def get_dataframes(self):
        return self.dataframes

    def get_the(self, df_key):
        return self.df[df_key]

    def get_color_palette(self):
        """
        Renvoie un dictionnaire de palettes de couleurs adaptées aux types de données.
        - 'qualitative': pour les données catégorielles, utilise des palettes de couleurs distinctes adaptées au daltonisme.
        - 'continuous': pour les données continues, utilise des palettes continues adaptées au daltonisme.
        - 'cyclical': pour les données périodiques, utilise des palettes cycliques.
        - 'sequential': pour les données ordonnées ou hiérarchiques, utilise des palettes séquentielles adaptées au daltonisme.
        """
        return {
            'qualitative': {  #(catégorielles)
                'D3': px.colors.qualitative.D3,     # color blindness ok
                'T10': px.colors.qualitative.T10,   # color blindness ok
                'Safe': px.colors.qualitative.Safe  # color blindness ok

            },
            'continuous': {  #(continues)
                'Viridis': px.colors.sequential.Viridis,  # color blindness ok
                'Cividis': px.colors.sequential.Cividis,  # color blindness ok
                'Inferno': px.colors.sequential.Inferno,  # color blindness ok
                'Magma': px.colors.sequential.Magma,      # color blindness ok
                'Plasma': px.colors.sequential.Plasma,    # color blindness ok
                'Turbo': px.colors.sequential.Turbo       # color blindness ok
            },
            'cyclical': {  #(périodiques)
                'IceFire': px.colors.cyclical.IceFire,  # color blindness (best option)
            },
            'sequential': { #(séquentielles)
                'Viridis': px.colors.sequential.Viridis,  # color blindness ok
                'Cividis': px.colors.sequential.Cividis,  # color blindness ok
                'Inferno': px.colors.sequential.Inferno,  # color blindness ok
                'Magma': px.colors.sequential.Magma,      # color blindness ok
                'Plasma': px.colors.sequential.Plasma,    # color blindness ok
                'Blues': px.colors.sequential.Blues,      # color blindness ok
                'Greys': px.colors.sequential.Greys       # color blindness ok
            }
        }

    ## --- 1) SCATTER ---
    def plot_scatter_with_color(self, df_key, x_col, y_col, color_col, palette_type, palette_name):
        """
        Tracer un graphique de dispersion avec des couleurs personnalisées.
        Paramètres :
        - df_key : clé du DataFrame.
        - x_col, y_col : colonnes pour les axes x et y.
        - color_col : colonne utilisée pour colorer les points.
        - palette_type : 'continuous' pour une palette continue ou 'categorical' pour une palette discrète.
        - palette_name : nom de la palette à utiliser.
        """
        df = self.df[df_key]
        color_palette = self.get_color_palette()

        if x_col and y_col and color_col:

            if palette_type == 'continuous':
                color_scale = color_palette['continuous'].get(palette_name, px.colors.cyclical.mrybm_r)
                color_discrete_map = None  # Pas de couleurs discrètes pour une palette continue
            else:
                color_scale = None  # Pas de couleur continue pour une palette discrète
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
        df = self.df[df_key]
        color_palette = self.get_color_palette()

        if palette_type not in color_palette:
            raise ValueError(f"Type de palette invalide. Choisissez parmi {list(color_palette.keys())}.")

        if palette_name not in color_palette[palette_type]:
            raise ValueError(
                f"Palette '{palette_name}' non trouvée pour le type '{palette_type}'. Choisissez parmi {list(color_palette[palette_type].keys())}.")

        palette = color_palette[palette_type][palette_name]
        color_map = None

        if palette_type == 'qualitative':
            # Pour les palettes qualitatives, `color_discrete_map` doit être un dictionnaire mappant les valeurs aux couleurs
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
        Tracer un pair plot à partir d'un DataFrame Pandas.
        Paramètres:
        - data: DataFrame contenant les données.
        - hue: Nom de la colonne pour colorer les points selon une variable catégorielle.
        - palette: Palette de couleurs à utiliser pour le graphique.
        """
        sns.set_theme(style='darkgrid', rc={
            'axes.facecolor': '#313234',    # Couleur de fond des axes
            'figure.facecolor': '#313234',  # Couleur de fond de la figure
            'axes.labelcolor': 'white',     # Couleur des labels des axes
            'xtick.color': 'white',         # Couleur des ticks de l'axe x
            'ytick.color': 'white',         # Couleur des ticks de l'axe y
            'grid.color': '#444444',        # Couleur de la grille
            'text.color': 'white'           # Couleur du texte
        })

        # 'Set1', 'Set2', 'Set3', 'Paired', 'Dark2', 'Pastel1', 'Pastel2', 'Accent', 'husl', 'hls' --> Qualitative

        fig = sns.pairplot(data, hue=hue, palette=palette)
        st.pyplot(fig)







    ## --- Histogram ---
    def plot_distribution_histogram(self, df_key, column_name, color='blue'):
        color = color or self.get_color_palette()['blue']
        df = self.df[df_key]

        fig = px.histogram(df, x=column_name, marginal="kde", color=color)

        fig.update_layout(
            title=f'Distribution of {column_name}',
            title_font_color='white',
            xaxis_title=column_name,
            xaxis_title_font_color='white',
            yaxis_title='Frequency',
            yaxis_title_font_color='white',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)





    ## --- Specific ---
    def plot_correlation_matrix(self, df_key):
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
            title='Matrice de Corrélation',
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
        df = self.df[df_key]

        numeric_df = df.select_dtypes(include=[float, int])

        if target_variable not in numeric_df.columns:
            st.error(f"La variable cible '{target_variable}' n'est pas présente dans le DataFrame.")
            return

        correlations = numeric_df.corr()[target_variable].dropna().sort_values(ascending=False)

        correlations = correlations[correlations.index != target_variable]

        col1, col2 = st.columns([1,3])
        with col1:
            st.write(f"### Corrélations avec '{target_variable}'")
            st.dataframe(correlations)
        with col2:
            fig = go.Figure(data=go.Bar(
                x=correlations.index,
                y=correlations.values,
                marker_color='#00d2ba'
            ))

            fig.update_layout(
                title=f'Coefficients de corrélation avec {target_variable}',
                xaxis_title='Variables',
                yaxis_title='Coefficient de Corrélation',
                plot_bgcolor='#313234',
                paper_bgcolor='#313234',
                font=dict(color='white')
            )

            st.plotly_chart(fig, use_container_width=True)



    def boxplot(self, df_key, x_col, y_col):
        """
        Trace un graphique avec plusieurs boxplots de x_col en fonction de y_col.

        Paramètres:
        - df_key : La clé du DataFrame à utiliser.
        - x_col : La colonne du DataFrame à utiliser pour l'axe x.
        - y_col : La colonne du DataFrame à utiliser pour l'axe y.
        """
        data = self.df[df_key]

        fig = plt.figure(figsize=(10, 6))  # Définir la taille de la figure
        sns.boxplot(x=x_col, y=y_col, data=data, hue=x_col, palette="Set2", legend=False)  # Créer le boxplot avec seaborn

        plt.xlabel(x_col)  # Label pour l'axe x
        plt.ylabel(y_col)  # Label pour l'axe y
        plt.title(f'Boxplot de {x_col} en fonction de {y_col}')  # Titre du graphique

        st.pyplot(fig)                               # Afficher le graphique

    def decompose_time_series(self, df_key, time_col, value_col, period=12):
        df = self.df[df_key]

        if time_col not in df.columns or value_col not in df.columns:
            st.error(f"Les colonnes '{time_col}' ou '{value_col}' n'existent pas dans le DataFrame.")
            return

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(by=time_col)

        df = df.dropna(subset=[time_col, value_col])

        df.set_index(time_col, inplace=True)

        # Effectuer la décomposition STL
        try:
            stl = STL(df[value_col], period=period)
            result = stl.fit()
        except Exception as e:
            st.error(f"Erreur lors de la décomposition STL: {e}")
            return

        # Tracer les résultats
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle(f'Décomposition de la série temporelle : {value_col}', fontsize=16)

        # Tendance
        axes[0].plot(result.trend, label='Tendance', color='blue')
        axes[0].set_title('Tendance')
        axes[0].legend(loc='upper left')

        # Saison
        axes[1].plot(result.seasonal, label='Saisonnalité', color='green')
        axes[1].set_title('Saisonnalité')
        axes[1].legend(loc='upper left')

        # Résidu
        axes[2].plot(result.resid, label='Résidu', color='red')
        axes[2].set_title('Résidu')
        axes[2].legend(loc='upper left')

        # Série originale
        axes[3].plot(df[value_col], label='Série Originale', color='gray')
        axes[3].set_title('Série Originale')
        axes[3].legend(loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        st.pyplot(fig)