import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Masking, Input, Dropout

class LSTMModel():
    def __init__(self, min_sequence_length=2, forecast_months=6):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.model = None  # Model initialisation to None

    def train(self, x_train, y_train):
        # Print the shape and type of training data
        print("Training input shape:", x_train.shape)
        print("Training input type:", type(x_train))
        print("Training target shapes:")
        print("lengths_filtered_output:", y_train['lengths_filtered_output'].shape)
        print("lengths_measured_output:", y_train['lengths_measured_output'].shape)

        # Correct input shape based on data dimensions
        input_shape = (self.min_sequence_length, x_train.shape[2])

        # Define model architecture
        inputs = Input(shape=input_shape)
        x = Masking(mask_value=0.0)(inputs)
        x = LSTM(64, return_sequences=False)(x)

        # Ajout de couches régressives
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)

        # Output layers for both predictions
        forecast_lengths_filtered = Dense(self.forecast_months, name='lengths_filtered_output')(x)
        forecast_lengths_measured = Dense(self.forecast_months, name='lengths_measured_output')(x)

        # Compile model
        self.model = Model(inputs=inputs, outputs=[forecast_lengths_filtered, forecast_lengths_measured])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'accuracy'])

        # Training settings
        num_epochs = 100
        batch_size = 32

        # Print check for NaN and inf values in training data
        print(np.isnan(x_train).any(), np.isinf(x_train).any())
        print(np.isnan(y_train['lengths_filtered_output']).any(), np.isinf(y_train['lengths_filtered_output']).any())
        print(np.isnan(y_train['lengths_measured_output']).any(), np.isinf(y_train['lengths_measured_output']).any())

        # Reshape y_train if necessary
        y_train_reshaped = {
            'lengths_filtered_output': np.array(y_train['lengths_filtered_output']),
            'lengths_measured_output': np.array(y_train['lengths_measured_output'])
        }

        # Training loop with progress bar in Streamlit
        with st.spinner('Training the model...'):
            progress_bar = st.progress(0)
            for epoch in range(num_epochs):
                self.model.fit(
                    x_train,
                    y_train_reshaped,
                    epochs=1,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=1
                )
                progress_bar.progress((epoch + 1) / num_epochs)
            progress_bar.empty()

    def predict(self, x_test):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        with st.spinner('Generating predictions...'):
            progress_bar = st.progress(0)
            num_batches = len(x_test) // 32 + (len(x_test) % 32 != 0)
            predictions = []

            for i in range(num_batches):
                batch = x_test[i * 32:(i + 1) * 32]
                batch_predictions = self.model.predict(batch)
                predictions.append(batch_predictions)
                progress_bar.progress((i + 1) / num_batches)

            # Flatten and concatenate predictions
            predictions_filtered, predictions_measured = zip(*predictions)
            predictions_filtered = np.concatenate(predictions_filtered, axis=0)
            predictions_measured = np.concatenate(predictions_measured, axis=0)

            progress_bar.empty()
            st.success('Predictions generated successfully!')

        return {'lengths_filtered_output': predictions_filtered, 'lengths_measured_output': predictions_measured}