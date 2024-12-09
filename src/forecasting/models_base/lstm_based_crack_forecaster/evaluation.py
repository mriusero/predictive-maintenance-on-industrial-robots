import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras import callbacks

from src.forecasting.validation.display import DisplayData


class TrainingPlot(callbacks.Callback):
    def __init__(self, output_names):
        super().__init__()
        self.output_names = output_names
        self.history = {name: {'loss': [], 'val_loss': [], 'metrics': [], 'val_metrics': []} for name in output_names}
        self.global_history = {'loss': [], 'val_loss': []}
        self.plot_placeholders = {name: st.empty() for name in output_names}

    def on_epoch_end(self, epoch, logs={}):
        self.global_history['loss'].append(logs.get('loss'))
        self.global_history['val_loss'].append(logs.get('val_loss'))

        for name in self.output_names:
            self.history[name]['loss'].append(logs.get(f'{name}_loss', 0))
            self.history[name]['val_loss'].append(logs.get(f'val_{name}_loss', 0))
            metric_key = f'{name}_accuracy'
            val_metric_key = f'val_{name}_accuracy'
            self.history[name]['metrics'].append(logs.get(metric_key, 0))
            self.history[name]['val_metrics'].append(logs.get(val_metric_key, 0))

        for name in self.output_names:
            if len(self.history[name]['loss']) > 1:
                epochs = np.arange(0, len(self.history[name]['loss']))
                plt.figure(figsize=(10, 5))
                plt.plot(epochs, self.history[name]['loss'], label="Train Loss", color="blue", linestyle="-")
                plt.plot(epochs, self.history[name]['val_loss'], label="Val Loss", color="blue", linestyle=":")
                plt.plot(epochs, self.history[name]['metrics'], label="Train Accuracy", color="red", linestyle="-")
                plt.plot(epochs, self.history[name]['val_metrics'], label="Val Accuracy", color="red", linestyle=":")
                plt.title(f"Training Metrics for Output: {name} [Epoch {epoch + 1}]")
                plt.xlabel("Epoch")
                plt.ylabel("Loss/Accuracy")
                plt.legend()

                self.plot_placeholders[name].pyplot(plt)
                plt.close()


def save_predictions(output_path, df, step):

    file_path = f"{output_path}/lstm_predictions_{step}.csv"
    df.to_csv(file_path, index=False)

    return f"Predictions saved successfully: {output_path}"


def display_results(df):
    display = DisplayData(df)

    col1, col2 = st.columns(2)
    with col1:
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'source')
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'item_id')
        display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'Fatigue crack')
    with col2:
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'source')
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'item_id')
        display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'Fatigue crack')

    display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'item_id')
    st.dataframe(df)
