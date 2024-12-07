import streamlit as st
import pandas as pd
import numpy as np

from src.forecasting import run_statistical_test
from src.core import display_variable_types


def page_1():
    st.markdown('<div class="header">#1 Exploration_</div>', unsafe_allow_html=True)

    st.markdown('## #Evolution of crack length over time_')
    st.markdown("""
    The plots illustrate the evolution of crack length over time. The left plot shows the raw measured crack lengths, while the right plot presents the filtered data to reduce noise. Each point represents a monthly measurement, categorized by different failure modes: `Infant mortality`, `Fatigue crack`, and `Control board failure`. The filtering highlights the underlying growth trends more clearly.
    """)
    col1, col2 = st.columns([2, 2])

    qualitative_pal = 'Safe'
    continuous_pal = 'Viridis'
    cyclical_pal = 'IceFire'
    sequential_pal = 'Inferno'

    with col1:
        st.session_state.data.plot_scatter_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_measured',
                                                      color_col='Failure mode',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)


    with col2:
        st.session_state.data.plot_scatter_with_color(df_key='train',
                                                      x_col='time (months)',
                                                      y_col='length_filtered',
                                                      color_col='Failure mode',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)

    st.markdown("""
        The histograms show the distribution of the cumulative crack length over time for different failure modes: `Infant mortality`, `Fatigue crack`, and `Control board failure`. The left plot displays the sum of measured lengths, while the right plot shows the sum after filtering to reduce noise. The box plots above each histogram summarize the distribution of crack lengths across the failure modes.
                """)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.data.plot_histogram_with_color(df_key='train',
                                                        x_col='time (months)',
                                                        y_col='length_measured',
                                                        color_col='Failure mode',
                                                        palette_type='qualitative',
                                                        palette_name=qualitative_pal)
    with col2:
        st.session_state.data.plot_histogram_with_color(df_key='train',
                                                        x_col='time (months)',
                                                        y_col='length_filtered',
                                                        color_col='Failure mode',
                                                        palette_type='qualitative',
                                                        palette_name=qualitative_pal)


    st.markdown("## #PSEUDO_TEST_WITH_TRUTH_")
    st.markdown('## #End of life_')
    st.markdown("""
This section visualizes the crack length progression over time in relation to labels and remaining useful life (RUL).  
* The top row shows scatter plots of measured and filtered crack lengths, colored by a classification label ranging from 0 (RUL > 6 months) to 1 (RUL < 6 months).  
* The bottom row presents similar plots, but colored according to the true RUL values.  
    
These visualizations help to understand that the end of life of a robot extends over a wide range from 1 to 50 months and that it is necessary to classify the type of failure to determine a more likely consistent range.    """)
    col1, col2 = st.columns([2, 2])
    with col1:
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'label',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'true_rul',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'label',
                                                        palette_type='qualitative',
                                                        palette_name=qualitative_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_measured', 'true_rul',
                                                        palette_type='qualitative',
                                                        palette_name=qualitative_pal)

    with col2:
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'label',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_scatter_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'true_rul',
                                                      palette_type='continuous',
                                                      palette_name=continuous_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'label',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)
        st.session_state.data.plot_histogram_with_color('pseudo_test_with_truth', 'time (months)', 'length_filtered', 'true_rul',
                                                      palette_type='qualitative',
                                                      palette_name=qualitative_pal)

    st.markdown("## #Statistics_")

    df = pd.read_csv('./data/output/training/training_data.csv')
    statistics = df.copy().describe(include=[np.number])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('#### #Describe_')
        st.dataframe(statistics)
    with col2:
        st.markdown('#### #Variables types_')
        variable_types_df = display_variable_types(df)
        st.dataframe(variable_types_df)

    st.markdown('#### #Failures stats_')
    grouped_stats_df = df.copy().groupby('Failure mode').describe()
    st.dataframe(grouped_stats_df)

    st.markdown("#### #Normality_")

    run_statistical_test(df, 'normality', 'time (months)')
    run_statistical_test(df, 'normality', 'crack length (arbitary unit)')
    run_statistical_test(df, 'normality', 'rul (months)')
    run_statistical_test(df, 'normality', 'Time to failure (months)')
