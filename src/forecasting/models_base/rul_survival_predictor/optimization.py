import os
import pickle

import json
import numpy as np
import optuna
import traceback
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from src.forecasting.models_base.rul_survival_predictor.configs import MODEL_FOLDER, HYPERPARAMETERS_PATH,  MODEL_NAME, N_VAL_SETS


def load_hyperparameters():
    """
    Loads the best hyperparameters from a JSON file.
    """
    if os.path.exists(HYPERPARAMETERS_PATH):
        with open(HYPERPARAMETERS_PATH, 'r') as f:
            best_params = json.load(f)  # Use json.load instead of pickle.load
        return best_params
    else:
        return None


def save_hyperparameters(best_params):
    """
    Saves the best hyperparameters to a JSON file.
    """
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(HYPERPARAMETERS_PATH, 'w') as f:
        json.dump(best_params, f, indent=4)


def save_study_logs(study):
    """
    Saves the Optuna study logs into a JSON file.
    """
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    study_logs = {
        "study_name": study.study_name,
        "direction": study.direction,
        "best_trial": study.best_trial.params,
        "all_trials": [
            {
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state
            }
            for trial in study.trials
        ]
    }
    log_file = os.path.join(MODEL_FOLDER, f"{MODEL_NAME}_optimisation_logs.json")
    with open(log_file, 'w') as f:
        json.dump(study_logs, f, indent=4)


def save_error_log(exception):
    """
    Saves the error log into a separate file for debugging.
    """
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    error_message = {
        "error": str(exception),
        "stack_trace": traceback.format_exc()
    }
    error_file = os.path.join(MODEL_FOLDER, "error_log.json")
    with open(error_file, 'w') as f:
        json.dump(error_message, f, indent=4)


def optimize_hyperparameters(x_train, y_train, validation_sets, percentile=None, n_trials=None):
    """
    Optimizes hyperparameters using Optuna, evaluating across multiple validation sets.
    """

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
        }

        model = GradientBoostingSurvivalAnalysis(**params)
        model.fit(x_train, y_train)

        ci_scores = []

        for i in range(N_VAL_SETS):
            x_val, y_val = validation_sets[f'validation_set_{i + 1}']

            survival_times = []
            for sf in model.predict_survival_function(x_val):
                if len(sf.x) == 0:
                    survival_times.append(np.nan)
                    continue
                idx = np.searchsorted(sf.y, percentile / 100.0, side='left')
                if idx >= len(sf.x):
                    survival_times.append(sf.x[-1])
                else:
                    survival_times.append(sf.x[idx] if idx < len(sf.x) else sf.x[-1])

            survival_times = [t if not np.isnan(t) else max(y_val['time (months)']) for t in survival_times]
            ci = concordance_index_censored(y_val['label'], y_val['time (months)'], np.array(survival_times))[0]
            ci_scores.append(ci)

        return np.mean(ci_scores)

    study = optuna.create_study(direction='maximize', study_name='rul_survival_predictor_hyperparameter_optimization')

    try:
        study.optimize(objective, n_trials=n_trials)

    except Exception as e:
        print(f"Optimization interrupted due to: {e}")
        traceback.print_exc()
        save_error_log(e)

    finally:
        if study.best_trial:
            save_hyperparameters(study.best_trial.params)
        save_study_logs(study)

    return study.best_params


