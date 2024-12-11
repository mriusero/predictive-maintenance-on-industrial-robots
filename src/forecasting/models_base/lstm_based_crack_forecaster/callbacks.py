import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tensorflow.keras import callbacks


class TrainingPlot(callbacks.Callback):
    def __init__(self, output_names):
        """
        Initializes the callback for plotting training metrics.
        :param output_names: List of output names to track metrics for
        """
        super().__init__()
        self.output_names = output_names
        self.history = {name: {'loss': [], 'val_loss': [], 'metrics': [], 'val_metrics': [],        # Initialize the history dictionary to store metrics for each output
                               'mae': [], 'mape': [], 'auc': [], 'val_auc': [],
                               'val_mae': [], 'val_mape': []} for name in output_names}
        self.global_history = {'loss': [], 'val_loss': []}
        self.plot_placeholders = {name: st.empty() for name in output_names}

        plt.style.use('dark_background')            # Set matplotlib style to dark mode

    def on_epoch_end(self, epoch, logs={}):
        """
        Updates the metrics history after each epoch and plots the graphs.
        :param epoch: The current epoch number
        :param logs: Dictionary containing the metric values for the epoch
        """
        self.global_history['loss'].append(logs.get('loss'))                # Update global loss values
        self.global_history['val_loss'].append(logs.get('val_loss'))

        for name in self.output_names:

            self.history[name]['loss'].append(logs.get(f'{name}_loss', 0))              # Update individual output metrics (loss, accuracy, etc.)
            self.history[name]['val_loss'].append(logs.get(f'val_{name}_loss', 0))
            metric_key = f'{name}_accuracy'
            val_metric_key = f'val_{name}_accuracy'
            self.history[name]['metrics'].append(logs.get(metric_key, 0))
            self.history[name]['val_metrics'].append(logs.get(val_metric_key, 0))

            self.history[name]['mae'].append(logs.get(f'{name}_mae', None))             # Add mae, mape, auc for training and validation
            self.history[name]['mape'].append(logs.get(f'{name}_mape', None))
            self.history[name]['auc'].append(logs.get(f'{name}_AUC', None))
            self.history[name]['val_mae'].append(logs.get(f'val_{name}_mae', None))
            self.history[name]['val_mape'].append(logs.get(f'val_{name}_mape', None))
            self.history[name]['val_auc'].append(logs.get(f'val_{name}_AUC', None))

        for name in self.output_names:

            if len(self.history[name]['loss']) > 1:

                epochs = np.arange(0, len(self.history[name]['loss']))
                fig, axs = plt.subplots(1, 2, figsize=(15, 5))          # Create a figure with two subplots side by side

                # Plot loss and accuracy on the first axis
                axs[0].plot(epochs, self.history[name]['loss'], label="Train Loss", color="blue", linestyle="-")
                axs[0].plot(epochs, self.history[name]['val_loss'], label="Val Loss", color="blue", linestyle=":")
                axs[0].plot(epochs, self.history[name]['metrics'], label="Train Accuracy", color="red", linestyle="-")
                axs[0].plot(epochs, self.history[name]['val_metrics'], label="Val Accuracy", color="red", linestyle=":")
                axs[0].set_title(f"Loss & Accuracy for Output: {name} [Epoch {epoch + 1}]")
                axs[0].set_xlabel("Epoch")
                axs[0].set_ylabel("Loss/Accuracy")
                axs[0].legend()

                # Plot MAE and MAPE with different y-axes
                if None not in self.history[name]['mae']:
                    axs[1].plot(epochs, self.history[name]['mae'], label="Train MAE", color="green", linestyle="-")
                    axs[1].plot(epochs, self.history[name]['val_mae'], label="Val MAE", color="green", linestyle=":")

                if None not in self.history[name]['mape']:
                    ax2 = axs[1].twinx()
                    ax2.plot(epochs, self.history[name]['mape'], label="Train MAPE", color="orange", linestyle="-")
                    ax2.plot(epochs, self.history[name]['val_mape'], label="Val MAPE", color="orange", linestyle=":")
                    ax2.set_ylabel("MAPE")
                    ax2.tick_params(axis='y', labelcolor="orange")
                    ax2.legend(loc='upper right')

                # Add AUC to a third vertical axis
                if None not in self.history[name]['auc']:
                    ax3 = axs[1].twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                    ax3.plot(epochs, self.history[name]['auc'], label="Train AUC", color="purple", linestyle="-")
                    ax3.plot(epochs, self.history[name]['val_auc'], label="Val AUC", color="purple", linestyle=":")
                    ax3.set_ylabel("AUC")
                    ax3.tick_params(axis='y', labelcolor="purple")
                    ax3.legend(loc='upper left')

                # Adjust the left y-axis for MAE
                axs[1].set_title(f"MAE, MAPE & AUC for Output: {name} [Epoch {epoch + 1}]")
                axs[1].set_xlabel("Epoch")
                axs[1].set_ylabel("MAE")
                axs[1].legend(loc='upper left')

                # Display the figure in Streamlit
                self.plot_placeholders[name].pyplot(fig)
                plt.close()