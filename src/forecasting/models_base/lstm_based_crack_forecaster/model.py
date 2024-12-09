#model.py
import os
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.layers import LSTM, Dense, Masking, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import plot_model
import visualkeras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from .configs import MIN_SEQUENCE_LENGTH, FORECAST_MONTHS, FEATURE_COLUMNS, MODEL_PATH, MODEL_FOLDER


class LSTMModel():
    def __init__(self, min_sequence_length=MIN_SEQUENCE_LENGTH, forecast_months=FORECAST_MONTHS):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.model = None

    def build_model(self, input_shape=(MIN_SEQUENCE_LENGTH, len(FEATURE_COLUMNS))):
        """
        Builds and returns a multitask deep learning model.
        Parameters:
            input_shape (tuple): Shape of the input data (timesteps, features).
        Returns:
            model (tf.keras.Model): Compiled Keras model.
        """
        input_layer = Input(shape=input_shape, name="Input_Layer")
        x = Masking(mask_value=0.0)(input_layer)
        x = LSTM(64, return_sequences=True, name="LSTM_Layer_1")(x)
        x = LSTM(128, return_sequences=False, name="LSTM_Layer_2")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Branche pour length_filtered
        x_length = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x_length = BatchNormalization()(x_length)
        x_length = Dropout(0.2)(x_length)
        x_length = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_length)
        output_length_filtered = Dense(6, activation='sigmoid', name='length_filtered')(x_length)
        output_length_measured = Dense(6, activation='sigmoid', name='length_measured')(x_length)

        # Branche pour la classification
        x_classification = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x_classification = BatchNormalization()(x_classification)
        x_classification = Dropout(0.2)(x_classification)
        x_classification = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_classification)
        output_infant_mortality = Dense(6, activation='softmax', name='Infant_mortality')(x_classification)                # Outputs for classification tasks
        output_control_board_failure = Dense(6, activation='softmax', name='Control_board_failure')(x_classification)
        output_fatigue_crack = Dense(6, activation='softmax', name='Fatigue_crack')(x_classification)

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
                'length_filtered': ['accuracy'],
                'length_measured': ['accuracy'],
                'Infant_mortality': ['accuracy'],
                'Control_board_failure': ['accuracy'],
                'Fatigue_crack': ['accuracy'],
            },
            loss_weights={
                'length_filtered': 1.0,
                'length_measured': 1.0,
                'Infant_mortality': 1.0,
                'Control_board_failure': 1.0,
                'Fatigue_crack': 1.0,
            }
        )
        self.model.summary()

        try:
            os.makedirs(MODEL_FOLDER, exist_ok=True)
            plot_model(self.model, to_file=f'{MODEL_FOLDER}' +'architecture.png', show_shapes=True, show_layer_names=True)

            stylized_img = visualkeras.layered_view(self.model, legend=True)
            stylized_img.save(f'{MODEL_FOLDER}' + 'model_stylized.png')

            fig, axs = plt.subplots(1, 2, figsize=(15, 20))

            img1 = mpimg.imread(f'{MODEL_FOLDER}' +'architecture.png')
            axs[0].imshow(img1)
            axs[0].axis('off')
            axs[0].set_title('Architecture')

            img2 = mpimg.imread(f'{MODEL_FOLDER}' + 'model_stylized.png')
            axs[1].imshow(img2)
            axs[1].axis('off')
            axs[1].set_title('Stylized Model')

            st.pyplot(fig)
        except Exception as e:
            st.error('Error while generating model architecture images.'
                     'You must install graphviz and pydot to generate the model architecture images.')

        return self.model


    def train(
            self, x_, y_,
            epochs=50, batch_size=32, validation_split=0.2,
            callbacks=None
        ):
        """
        Trains the model on the provided training data.
        Parameters:
            x_ (np.ndarray): Training data of shape (num_samples, timesteps, features).
            y_ (dict): Dictionary containing training targets for each output of the model.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Number of samples per gradient update.
            validation_split (float): Fraction of the training data to be used as validation data.
        Returns:
            tf.keras.callbacks.History: Training history.
        """
        if self.model is None:
            self.model = self.build_model()

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
        Generates predictions for the provided test data using the trained model.
        Parameters:
            x_test (np.ndarray): Test data of shape (num_samples, timesteps, features).
        Returns:
            dict: A dictionary containing the predictions for each model output.
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été formé.")

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @staticmethod
    def load_model(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)