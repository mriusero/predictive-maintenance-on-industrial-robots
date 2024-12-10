import numpy as np
import optuna
import os
import json
import pickle
import traceback
import tensorflow as tf

from .evaluation import generate_model_images
from .configs import MODEL_FOLDER

def optimize_hyperparameters(lstm, x_, y_, n_trials=50, path="best_params.pkl", log_path="study_logs.json"):
    """
    Optimizes hyperparameters using Optuna with a focus on specific metrics and early stopping.
    Parameters:
        lstm (LSTMModel): LSTMModel instance for training.
        x_ (np.ndarray): Training data of shape (num_samples, timesteps, features).
        y_ (dict): Dictionary containing training targets for each output of the model.
        n_trials (int): Number of Optuna trials for optimization.
        path (str): Path to save the best hyperparameters.
        log_path (str): Path to save the study logs.
    """

    def objective(trial):
        """
        Objective function for Optuna optimization.
        """
        lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128, step=16)      # Define the search space
        lstm_units_2 = trial.suggest_int('lstm_units_2', 64, 256, step=16)
        lstm_units_3 = trial.suggest_int('lstm_units_3', 128, 512, step=32)
        dense_units = trial.suggest_int('dense_units', 32, 256, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        lstm.model = lstm.build_model(                  # Build the model with the suggested parameters
            input_shape=(x_.shape[1], x_.shape[2]), optimize=True)

        lstm.model.compile(                                                         # Compile the model
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
                'Control_board_failure': 1.5,
                'Fatigue_crack': 1.0,
            }
        )

        generate_model_images(lstm.model, MODEL_FOLDER)

        early_stopping = tf.keras.callbacks.EarlyStopping(                  # Implement early stopping
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        history = lstm.model.fit(           # Train the model
            x_,
            {
                'length_filtered': y_['length_filtered'],
                'length_measured': y_['length_measured'],
                'Infant_mortality': y_['Infant mortality'],
                'Control_board_failure': y_['Control board failure'],
                'Fatigue_crack': y_['Fatigue crack'],
            },
            epochs=100,                         # Extended epochs for deeper exploration
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        val_metrics = history.history                   # Extract metrics to optimize
        mae_loss = np.mean(val_metrics['val_length_filtered_mae']) + np.mean(val_metrics['val_length_measured_mae'])
        mape_loss = np.mean(val_metrics['val_length_filtered_mape']) + np.mean(val_metrics['val_length_measured_mape'])
        auc_score = (
                np.mean(val_metrics['val_Infant_mortality_AUC']) +
                np.mean(val_metrics['val_Control_board_failure_AUC']) +
                np.mean(val_metrics['val_Fatigue_crack_AUC'])
        )
        return (mae_loss + mape_loss) - auc_score                   # Objective: Minimize MAE/MAPE and maximize AUC

    study = optuna.create_study(direction="minimize", study_name="lstm_hyperparameters")

    try:
        study.optimize(objective, n_trials=n_trials)
    except Exception as e:
        print(f"Optimization interrupted due to: {e}")
        traceback.print_exc()
    finally:
        os.makedirs(os.path.dirname(path), exist_ok=True)                   # Save the best hyperparameters
        with open(path, 'wb') as f:
            pickle.dump(study.best_params, f)
        print(f"Best hyperparameters saved to {path}: {study.best_params}")

        study_logs = study.trials_dataframe().to_dict(orient='records')     # Save the study logs in JSON format
        with open(log_path, 'w') as log_file:
            json.dump(study_logs, log_file, indent=4)
        print(f"Study logs saved to {log_path}")