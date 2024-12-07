import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from .optimization import load_hyperparameters


class GradientBoostingSurvivalModel:
    def __init__(self):
        self.model = None
        self.best_params = load_hyperparameters()


    def train(self, x_train, y_train):
        """
        Trains the gradient boosting survival model.
        """
        if not self.best_params:
            print('WARNING: default value for best_params !')
            self.best_params = {
                'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3,
                'subsample': 1.0, 'min_samples_split': 2, 'min_samples_leaf': 1
            }
        self.model = GradientBoostingSurvivalAnalysis(**self.best_params)
        self.model.fit(x_train, y_train)


    def predict(
            self,
            X_test: pd.DataFrame,
            columns_to_include: Optional[List[str]] = None,
            threshold: float = 0.5
            ) -> pd.DataFrame:
        """
        Makes predictions on the test data.
        """
        if not self.model:
            raise ValueError("The model has not been trained.")

        x_test_filtered = X_test[columns_to_include] if columns_to_include else X_test
        surv_funcs = self.model.predict_survival_function(x_test_filtered)

        predictions_df = X_test.copy()
        predictions_df['predicted_failure_now'] = np.nan
        predictions_df['predicted_failure_6_months'] = np.nan

        for idx, surv_func in enumerate(surv_funcs):
            time_now = X_test.loc[idx, 'time (months)']
            survival_prob_now = surv_func(time_now) if time_now <= surv_func.x[-1] else 1.0
            survival_prob_6_months = surv_func(time_now + 6) if time_now + 6 <= surv_func.x[-1] else 1.0

            predictions_df.at[idx, 'predicted_failure_now'] = 1 - survival_prob_now
            predictions_df.at[idx, 'predicted_failure_6_months'] = 1 - survival_prob_6_months

        predictions_df['predicted_failure_now_binary'] = (predictions_df['predicted_failure_now'] >= threshold).astype(
            int)
        predictions_df['predicted_failure_6_months_binary'] = (
                    predictions_df['predicted_failure_6_months'] >= threshold).astype(int)

        return predictions_df


    @staticmethod
    def compute_deducted_rul(group):
        """
        Computes the deducted RUL for each group.
        """
        index_failure = group[group['predicted_failure_now_binary'] == 1].index.min()

        if pd.isna(index_failure):
            return [0] * len(group)

        index_failure -= group.index.min()
        return list(range(index_failure, 0, -1)) + [1] + [0] * (len(group) - index_failure - 1)


    @staticmethod
    def save_predictions(model_name, submission_path, step, predictions_df):
        """
        Saves predictions to a CSV file with 'item_id' included.
        """
        file_path = f"{submission_path}/{model_name}/{model_name}_{step}.csv"
        predictions_df.to_csv(file_path, index=False)


    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @staticmethod
    def load_model(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
