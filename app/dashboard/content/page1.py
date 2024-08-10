import streamlit as st
import pandas as pd
import numpy as np

from ..functions import dataframing_data, load_data
from ..components import plot_correlation_matrix, plot_scatter, plot_scatter2, plot_histogram

def page_1():
    st.markdown('<div class="title">MACHINE LEARNING</div>', unsafe_allow_html=True)                            #TITLE
    st.markdown('<div class="header">#1 Exploration_</div>', unsafe_allow_html=True)

    dataframes = dataframing_data()

    texte = """
# Exploration_
        """
    st.markdown(texte)

    st.markdown('## TRAIN')
    col1, col2 = st.columns([2, 2])
    df = dataframes['train'].sort_values(by=['item_index', 'time (months)'])
    with col1:
        plot_scatter(df, 'time (months)', 'crack length (arbitary unit)', 'Failure mode')
        plot_correlation_matrix(df)
    with col2:
        plot_histogram(df, 'Failure mode')

    st.markdown('## PSEUDO_TEST_WITH_TRUTH')

    update_message = ''
    if st.button("Generate Random Data"):
        update_message = load_data()

    st.write(update_message)
    col1, col2 = st.columns([2, 2])
    df = dataframes['pseudo_test_with_truth'].sort_values(by=['item_index', 'time (months)'])
    with col1:
        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'label')
        df['RUL_inf_6'] = np.where(df['true_rul'] - df['time (months)'] <= 6, 1, 0)
        plot_scatter2(df, 'time (months)', 'crack length (arbitary unit)', 'RUL_inf_6')

    with col2:
        plot_histogram(df, 'label')
        plot_histogram(df, 'RUL_inf_6')
    st.write("Statistiques Descriptives :")  # Calcul des statistiques descriptives
    statistics = df.describe(include='all')
    st.dataframe(statistics)

    st.markdown('## TEST')
    df = dataframes['test'].sort_values(by=['item_index', 'time (months)'])
    plot_scatter(df, 'time (months)', 'crack length (arbitary unit)', 'item_index')
