import optuna
import numpy as np
import pickle
import os

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from ..configs import MODEL_FOLDER, HYPERPARAMETERS_PATH

def load_hyperparameters():
    """
    Loads the best hyperparameters from a file.
    """
    if os.path.exists(HYPERPARAMETERS_PATH):
        with open(HYPERPARAMETERS_PATH, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
    else:
        return None


def save_hyperparameters(best_params):
    """
    Saves the best hyperparameters to a file.
    """
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    with open(HYPERPARAMETERS_PATH, 'wb') as f:
        pickle.dump(best_params, f)


def optimize_hyperparameters(x_train, y_train):
    """
    Optimizes hyperparameters using Optuna.
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
        try:
            median_times = []
            for sf in model.predict_survival_function(x_train):
                if len(sf.x) == 0:
                    median_times.append(np.nan)
                    continue
                idx = np.searchsorted(sf.y, 0.5, side='left')
                if idx >= len(sf.x):
                    median_times.append(sf.x[-1])
                else:
                    median_times.append(sf.x[idx] if idx < len(sf.x) else sf.x[-1])
            median_times = [t if not np.isnan(t) else max(y_train['time (months)']) for t in median_times]

        except IndexError as e:
            print(f"IndexError encountered: {e}")
            for sf in model.predict_survival_function(x_train):
                print(f"sf.x: {sf.x}, sf.y: {sf.y}")
            raise e

        ci = concordance_index_censored(y_train['label'], y_train['time (months)'], np.array(median_times))[0]
        return ci

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    save_hyperparameters(study.best_params)

    return study.best_params