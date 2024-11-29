import os
import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, precision_recall_curve
)
import seaborn as sns

from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import mean_absolute_error
import optuna
import pickle

from ..display import DisplayData
from ..validation import generate_submission_file, calculate_score


class GradientBoostingSurvivalModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.best_params = None

    def prepare_data(
            self,
            df: pd.DataFrame,
            reference_df: Optional[pd.DataFrame] = None,
            columns_to_include: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepares the data for training and prediction.
        """
        df = df.sort_values(by=['item_id', 'time (months)']).reset_index(drop=True)

        if reference_df is not None:
            df = df.reindex(columns=reference_df.columns, fill_value=0)

        df['label'] = df['label'].astype(bool)
        df['time (months)'] = pd.to_numeric(df['time (months)'], errors='coerce')

        df.dropna(subset=['time (months)', 'label'], inplace=True)
        time_points = df[['item_id', 'time (months)']].drop_duplicates().sort_values(by=['item_id', 'time (months)'])

        if columns_to_include:
            X_prepared = df[columns_to_include]
        else:
            X_prepared = df.copy()

        try:
            y = Surv.from_dataframe('label', 'time (months)', X_prepared)
        except ValueError as e:
            print(f"Error creating survival object: {e}")
            raise e

        return X_prepared, y

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """
        Trains the gradient boosting survival model.
        """
        if not self.best_params:
            self.best_params = {
                'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 3,
                'subsample': 1.0, 'min_samples_split': 2, 'min_samples_leaf': 1
            }
        self.model = GradientBoostingSurvivalAnalysis(**self.best_params)
        self.model.fit(X_train, y_train)

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

        X_test_filtered = X_test[columns_to_include] if columns_to_include else X_test
        surv_funcs = self.model.predict_survival_function(X_test_filtered)

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

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: np.ndarray):
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
            model.fit(X_train, y_train)

            try:
                median_times = []
                for sf in model.predict_survival_function(X_train):
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
                for sf in model.predict_survival_function(X_train):
                    print(f"sf.x: {sf.x}, sf.y: {sf.y}")
                raise e

            ci = concordance_index_censored(y_train['label'], y_train['time (months)'], np.array(median_times))[0]
            return ci

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25)
        self.best_params = study.best_params
        self._save_hyperparameters()

    def _save_hyperparameters(self):
        """
        Saves the best hyperparameters to a file.
        """
        os.makedirs('data/output/submission/gradient_boosting', exist_ok=True)
        with open('data/output/submission/gradient_boosting/best_hyperparameters.pkl', 'wb') as f:
            pickle.dump(self.best_params, f)

    def _load_hyperparameters(self):
        """
        Loads the best hyperparameters from a file.
        """
        path = 'data/output/submission/gradient_boosting/best_hyperparameters.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.best_params = pickle.load(f)

    def evaluate_model_performance(self, true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
        """
        Evaluates the performance of the model based on true and predicted survival.
        """
        mae = mean_absolute_error(true_df['true_rul'], pred_df['predicted_failure_6_months'])
        st.write(f"MAE for 6-month survival predictions: {mae}")
        return mae

    def save_predictions(self, output_path, predictions_df, step):
        """
        Saves predictions to a CSV file with 'item_id' included.
        """
        file_path = f"{output_path}/gbsa_predictions_{step}.csv"
        predictions_df.to_csv(file_path, index=False)

    def display_results(self, X_train, predictions_df):
        """
        Displays results in Streamlit.
        """
        display = DisplayData(predictions_df)
        col1, col2 = st.columns(2)
        with col1:
            st.write(X_train.columns.to_list())
        with col2:
            self.plot_feature_importance(X_train)
        self.measure_performance_and_plot(predictions_df)

    def plot_feature_importance(self, X_train):
        """
        Plots feature importance of the trained model.
        """
        if not self.model:
            raise ValueError("The model has not been trained.")
        feature_importances = self.model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
        st.pyplot(plt)

    def compute_deducted_rul(self, group):
        """
        Computes the deducted RUL for each group.
        """
        index_failure = group[group['predicted_failure_now_binary'] == 1].index.min()

        if pd.isna(index_failure):
            return [0] * len(group)

        index_failure -= group.index.min()
        return list(range(index_failure, 0, -1)) + [1] + [0] * (len(group) - index_failure - 1)

    def measure_performance_and_plot(self, predictions_df: pd.DataFrame):
        """
        Measures model performance and plots relevant metrics.
        """
        true_labels = predictions_df['label_y']
        predicted_labels = predictions_df['predicted_failure_6_months_binary']

        # Classification Report
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())
        print(pd.DataFrame(report).transpose())

        col1, col2, col3 = st.columns(3)

        with col1:
            # Confusion Matrix
            cm = confusion_matrix(true_labels, predicted_labels)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
            cm_display.plot(cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            st.pyplot(plt)

        with col2:
            # ROC Curve
            fpr, tpr, _ = roc_curve(true_labels, predictions_df['predicted_failure_6_months'])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            st.pyplot(plt)

        with col3:
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(true_labels, predictions_df['predicted_failure_6_months'])
            plt.figure()
            plt.plot(recall, precision, color='b', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            st.pyplot(plt)

    def run_full_pipeline(
            self,
            train_df: pd.DataFrame,
            pseudo_test_with_truth_df: pd.DataFrame,
            test_df: pd.DataFrame,
            optimize: bool,
            columns_to_include: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs the full pipeline including training, validation, and testing with the option to include specific columns.
        """
        model_name = 'GradientBoostingSurvivalModel'
        output_path = 'data/output/submission/gradient_boosting'
        st.markdown("## GradientBoostingSurvivalModel Pipeline")
        print('Running GradientBoostingSurvivalModel Pipeline ...')

        # Prepare training data with column selection
        X_train, y_train = self.prepare_data(train_df.copy(), columns_to_include=columns_to_include)
        X_val, y_val = self.prepare_data(pseudo_test_with_truth_df.copy().drop(columns=['true_rul']), train_df.copy(),
                                         columns_to_include=columns_to_include)

        # Load or optimize hyperparameters
        self._load_hyperparameters()
        if optimize or not self.best_params:
            self.optimize_hyperparameters(X_train, y_train)
            st.success("Hyperparameters have been optimized and saved.")

        # Train the model with prepared sequential data
        self.train(X_train, y_train)

        # Validation
        val_predictions = self.predict(X_val, columns_to_include=columns_to_include)

        val_predictions['unique_index'] = val_predictions.apply(
            lambda row: f"id_{row['item_id']}_&_mth_{row['time (months)']}", axis=1
        )
        val_predictions.reset_index(drop=True, inplace=True)
        val_predictions.set_index('unique_index', inplace=True)
        pseudo_test_with_truth_df['unique_index'] = pseudo_test_with_truth_df.apply(
            lambda row: f"id_{row['item_id']}_&_mth_{row['time (months)']}", axis=1
        )
        pseudo_test_with_truth_df.reset_index(drop=True, inplace=True)
        pseudo_test_with_truth_df.set_index('unique_index', inplace=True)
        val_predictions_merged = pd.merge(val_predictions, pseudo_test_with_truth_df[['label', 'true_rul']],
                                          on='unique_index', how='left')

        val_predictions_merged.reset_index(drop=True, inplace=True)

        val_predictions_merged['deducted_rul'] = val_predictions_merged.groupby('item_id', group_keys=False).apply(
            self.compute_deducted_rul
        ).explode().astype(int).reset_index(drop=True)

        self.save_predictions('data/output/submission/gradient_boosting', val_predictions_merged, step='cross-val')
        generate_submission_file(model_name, output_path, step='cross-val')
        score = calculate_score('data/output/submission/gradient_boosting', step='cross-val')
        st.write(f"Cross-validation score: {score}")
        self.display_results(X_train, val_predictions_merged.sort_values(['item_id', 'time (months)'], ascending=True))

        # Final predictions
        X_test, _ = self.prepare_data(test_df.copy(), train_df.copy(), columns_to_include=columns_to_include)
        test_predictions = self.predict(X_test, columns_to_include=columns_to_include)
        test_predictions['deducted_rul'] = test_predictions.groupby('item_id', group_keys=False).apply(
            self.compute_deducted_rul
        ).explode().astype(int).reset_index(drop=True)

        self.save_predictions('data/output/submission/gradient_boosting', test_predictions, step='final-test')
        generate_submission_file(model_name, output_path, step='final-test')
        final_score = calculate_score('data/output/submission/gradient_boosting', step='final-test')
        st.write(f"Final score: {final_score}")

        return val_predictions_merged, test_predictions
