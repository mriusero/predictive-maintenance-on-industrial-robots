import pandas as pd
import streamlit as st

from src.forecasting import run_statistical_test


def page_2():                          #TITLE
    st.markdown('<div class="header">#2 Feature Engineering_</div>', unsafe_allow_html=True)
    texte = """

        Here is the Feature Engineering phase.

    """
    st.markdown(texte)

    st.markdown('## #Distributions_')
    st.session_state.data.plot_pairplot(data=pd.read_csv('./data/output/training/training_data.csv'),
                                        hue='Failure mode',
                                        palette='hls')

    df1 = pd.read_csv('./data/output/training/training_data.csv')
    df2 = st.session_state.data.get_the('train')

    # Obtenir les colonnes des deux DataFrames
    cols_df1 = set(df1.columns)
    cols_df2 = set(df2.columns)

    # Colonnes présentes dans df1 mais pas dans df2
    cols_only_in_df1 = cols_df1 - cols_df2
    st.write("Colonnes présentes uniquement dans le premier DataFrame :", cols_only_in_df1)

    # Colonnes présentes dans df2 mais pas dans df1
    cols_only_in_df2 = cols_df2 - cols_df1
    st.write("Colonnes présentes uniquement dans le second DataFrame :", cols_only_in_df2)

    st.session_state.data.plot_correlation_matrix(df_key='train')
    st.session_state.data.plot_correlation_with_target(df_key='train',
                                                       target_variable='label')
    st.session_state.data.plot_correlation_with_target(df_key='train',
                                                       target_variable='Time to failure (months)')
    st.session_state.data.plot_correlation_with_target(df_key='train',
                                                       target_variable='length_filtered')
    run_statistical_test(df2, 'multicollinearity')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### #Length Measured")
        st.session_state.data.decompose_time_series('train', 'time (months)', 'length_measured')
    with col2:
        st.markdown("### #Length Filtered")
        st.session_state.data.decompose_time_series('train', 'time (months)', 'length_filtered')