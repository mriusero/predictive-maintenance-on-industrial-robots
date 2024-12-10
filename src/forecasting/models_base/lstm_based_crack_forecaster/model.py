import os
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from keras.src.layers import BatchNormalization
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Masking,
    Input,
    Dropout,
    Bidirectional
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from .configs import (
    MIN_SEQUENCE_LENGTH,
    FORECAST_MONTHS,
    FEATURE_COLUMNS,
    MODEL_PATH,
    MODEL_FOLDER,
    HYPERPARAMETERS_PATH,
    LOG_PATH
)
from .evaluation import generate_model_images
from .optimization import optimize_hyperparameters


class LSTMModel:
    def __init__(self, min_sequence_length=MIN_SEQUENCE_LENGTH, forecast_months=FORECAST_MONTHS):
        """
        Initialize the LSTM model with default or provided configurations.
        """
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.model = None


    def build_model(self, input_shape=(MIN_SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), optimize=False):
        """
        Builds and returns a multitask deep learning model.
        Parameters:
            input_shape (tuple): Shape of the input data (timesteps, features).
            optimize (bool): If True, optimize hyperparameters before training.
        Returns:
            tf.keras.Model: Compiled Keras model.
        """
        input_layer = Input(shape=input_shape, name="Input_Layer")
        masked_input = Masking(mask_value=0.0)(input_layer)

        # Bidirectional LSTM layers to capture temporal dependencies
        x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(masked_input)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
        x = Bidirectional(LSTM(256, return_sequences=False, dropout=0.3))(x)

        # Dense layers for regression branch
        x_reg = Dense(256, activation='elu', kernel_regularizer=l2(0.01))(x)
        x_reg = BatchNormalization()(x_reg)
        x_reg = Dropout(0.3)(x_reg)
        x_reg = Dense(128, activation='elu', kernel_regularizer=l2(0.01))(x_reg)
        x_reg = Dropout(0.2)(x_reg)
        x_reg = Dense(64, activation='elu', kernel_regularizer=l2(0.01))(x_reg)
        x_reg = Dropout(0.2)(x_reg)
        x_reg = Dense(32, activation='elu', kernel_regularizer=l2(0.005))(x_reg)
        x_reg = Dropout(0.2)(x_reg)
        x_reg = Dense(16, activation='elu', kernel_regularizer=l2(0.005))(x_reg)
        x_reg = Dropout(0.1)(x_reg)

        output_length_filtered = Dense(6, activation='relu', name='length_filtered')(x_reg)
        output_length_measured = Dense(6, activation='relu', name='length_measured')(x_reg)

        # Dense layers for classification branch
        x_class = Dense(256, activation='elu', kernel_regularizer=l2(0.01))(x)
        x_class = BatchNormalization()(x_class)
        x_class = Dropout(0.3)(x_class)
        x_class = Dense(128, activation='elu', kernel_regularizer=l2(0.01))(x_class)
        x_class = Dropout(0.3)(x_class)
        x_class = Dense(64, activation='elu', kernel_regularizer=l2(0.01))(x_class)
        x_class = Dropout(0.2)(x_class)
        x_class = Dense(32, activation='elu', kernel_regularizer=l2(0.005))(x_class)
        x_class = Dropout(0.1)(x_class)
        x_class = Dense(16, activation='elu', kernel_regularizer=l2(0.005))(x_class)
        x_class = Dropout(0.1)(x_class)

        output_infant_mortality = Dense(6, activation='softmax', name='Infant_mortality')(x_class)
        output_control_board_failure = Dense(6, activation='softmax', name='Control_board_failure')(x_class)
        output_fatigue_crack = Dense(6, activation='softmax', name='Fatigue_crack')(x_class)

        # Compile final model
        self.model = Model(
            inputs=input_layer,
            outputs=[
                output_length_filtered,
                output_length_measured,
                output_infant_mortality,
                output_control_board_failure,
                output_fatigue_crack
            ]
        )
        self.model.compile(
            optimizer='adam',
            loss={
                'length_filtered': 'mse',
                'length_measured': 'mse',
                'Infant_mortality': 'binary_crossentropy',
                'Control_board_failure': 'binary_crossentropy',
                'Fatigue_crack': 'binary_crossentropy',
            },
            metrics={
                'length_filtered': ['mae', 'mape', 'accuracy'],
                'length_measured': ['mae', 'mape', 'accuracy'],
                'Infant_mortality': ['accuracy', 'AUC'],
                'Control_board_failure': ['accuracy', 'AUC'],
                'Fatigue_crack': ['accuracy', 'AUC'],
            },
            loss_weights={
                'length_filtered': 1.0,
                'length_measured': 1.0,
                'Infant_mortality': 1.0,
                'Control_board_failure': 1.0,
                'Fatigue_crack': 1.0,
            }
        )
        if not optimize:
            self.model.summary()
            generate_model_images(self.model, MODEL_FOLDER)

        return self.model


    def apply_best_params(self, best_params, optimize=False):
        """
        Configures the model with the best hyperparameters.
        Parameters:
            best_params (dict): Dictionary containing the best hyperparameters.
            optimize (bool): If True, optimize hyperparameters before training.
        """
        self.model = self.build_model(optimize=optimize)
        for layer in self.model.layers:
            if isinstance(layer, LSTM) or isinstance(layer, Dense):
                if 'units' in best_params:
                    layer.units = best_params.get(f"{layer.name}_units", layer.units)
            if isinstance(layer, Dropout):
                layer.rate = best_params.get('dropout_rate', layer.rate)

        learning_rate = best_params.get('learning_rate', 1e-3)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'length_filtered': 'mse',
                'length_measured': 'mse',
                'Infant_mortality': 'binary_crossentropy',
                'Control_board_failure': 'binary_crossentropy',
                'Fatigue_crack': 'binary_crossentropy',
            },
            metrics={
                'length_filtered': ['mae', 'mape', 'accuracy'],
                'length_measured': ['mae', 'mape', 'accuracy'],
                'Infant_mortality': ['accuracy', 'AUC'],
                'Control_board_failure': ['accuracy', 'AUC'],
                'Fatigue_crack': ['accuracy', 'AUC'],
            },
            loss_weights={
                'length_filtered': 1.0,
                'length_measured': 1.0,
                'Infant_mortality': 1.0,
                'Control_board_failure': 1.0,
                'Fatigue_crack': 1.0,
            }
        )


    def train(self, x_, y_, epochs=50, batch_size=32, validation_split=0.2, callbacks=None,
              optimize=False, n_trials=50, params_path=HYPERPARAMETERS_PATH, log_path=LOG_PATH):
        """
        Trains the model on the provided training data.

        Parameters:
            x_ (np.ndarray): Training data (samples, timesteps, features).
            y_ (dict): Dictionary containing targets for each model output.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            validation_split (float): Fraction of validation data.
            callbacks (list): Keras callbacks.
            optimize (bool): Optimize hyperparameters before training.
            n_trials (int): Number of trials for hyperparameter optimization.
            params_path (str): Path to save best hyperparameters.
            log_path (str): Path to save optimization logs.

        Returns:
            tf.keras.callbacks.History: Training history.
        """
        if optimize:
            print("Starting hyperparameter optimization...")
            optimize_hyperparameters(self, x_, y_, n_trials=n_trials, path=params_path, log_path=log_path)

        try:
            if os.path.exists(params_path):
                print(f"Loading best hyperparameters from {params_path}...")
                with open(params_path, 'rb') as f:
                    best_params = pickle.load(f)
                self.apply_best_params(best_params, optimize=optimize)
        except Exception as e:
            print(f"No hyperparameters found: {e}")
        finally:
            if self.model is None:
                print("Building model with default hyperparameters...")
                self.model = self.build_model(optimize=optimize)

        history = self.model.fit(
            x_,
            {
                'length_filtered': y_['length_filtered'],
                'length_measured': y_['length_measured'],
                'Infant_mortality': y_['Infant mortality'],
                'Control_board_failure': y_['Control board failure'],
                'Fatigue_crack': y_['Fatigue crack'],
            },
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks
        )
        self.save_model(path=MODEL_PATH)
        return history


    def predict(self, x_test):
        """
        Generates predictions for the provided test data.
        Parameters:
            x_test (np.ndarray): Test data (samples, timesteps, features).
        Returns:
            dict: Dictionary containing predictions for each model output.
        """
        if self.model is None:
            raise ValueError("The model has not been trained.")

        with st.spinner('Generating predictions...'):
            num_batches = len(x_test) // 32 + (len(x_test) % 32 != 0)

            all_predictions = {
                'length_filtered': [],
                'length_measured': [],
                'Infant_mortality': [],
                'Control_board_failure': [],
                'Fatigue_crack': [],
            }

            for i in range(num_batches):
                batch = x_test[i * 32:(i + 1) * 32]
                batch_predictions = self.model.predict(batch)

                for idx, key in enumerate(all_predictions.keys()):
                    all_predictions[key].append(batch_predictions[idx])

            for key in all_predictions.keys():
                all_predictions[key] = np.concatenate(all_predictions[key], axis=0)

        return all_predictions


    def save_model(self, path: str):
        """
            Saves the model to the specified path.
            Parameters:
                path (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        st.info(f"Model saved to {path}")


    @staticmethod
    def load_model(path: str):
        """
           Loads a saved model from the specified path.
           Parameters:
               path (str): Path to the saved model.
           Returns:
               LSTMModel: Loaded model instance.
           """
        with open(path, 'rb') as f:
            return pickle.load(f)