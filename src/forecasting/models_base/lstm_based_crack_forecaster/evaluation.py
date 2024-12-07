import streamlit as st
from src.forecasting.validation.display import DisplayData

def save_predictions(output_path, df, step):

    file_path = f"{output_path}/lstm_predictions_{step}.csv"
    df.to_csv(file_path, index=False)

    return f"Predictions saved successfully: {output_path}"


def display_results(self, df):
    display = DisplayData(df)

    col1, col2 = st.columns(2)
    with col1:
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'source')
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'item_id')
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'crack_failure')
    with col2:
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'source')
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'item_id')
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'crack_failure')

    # Remove the line that plots based on Failure mode as it's not used in V4
    # plot_scatter2(df, 'time (months)', 'length_measured', 'Failure mode (lstm)')