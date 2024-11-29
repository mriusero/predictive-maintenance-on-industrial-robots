import pandas as pd
from src.functions import load_failures
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from ..validation import generate_submission_file, calculate_score

class RandomForestClassifierModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.failure_mode_encoder = LabelEncoder()

    def fit_failure_mode_encoder(self, failures_df):
        failures_df.fillna({'Failure mode': 'UNKNOWN'}, inplace=True)
        failure_modes = failures_df['Failure mode']
        return self.failure_mode_encoder.fit(failure_modes)

    def encode_failure_mode(self, failure_modes):
        return self.failure_mode_encoder.transform(failure_modes)

    def decode_failure_mode(self, encoded_modes):
        return self.failure_mode_encoder.inverse_transform(encoded_modes)

    def prepare_data(self, df, target_col='Failure mode'):
        """Prepare data for training and prediction."""

        aggregated_df = df.groupby('item_id').agg({
            'time (months)': 'mean',
            'length_filtered': ['mean', 'std', 'max', 'min'],
            'length_measured': ['mean', 'std', 'max', 'min'],
            'source': 'mean',  # Mode for categorical data
        }).reset_index()

        # Flatten the column MultiIndex
        aggregated_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aggregated_df.columns.values]

        X = aggregated_df.drop(columns=['item_id'])

        failures_df = load_failures()
        self.fit_failure_mode_encoder(failures_df)

        if target_col in df.columns:
            y = df.groupby('item_id')[target_col].first().fillna('UNKNOWN')  # Aggregate targets
            y_encoded = self.encode_failure_mode(y)
            return X, y_encoded
        else:
            return X, None

    def train(self, X_train, y_train):
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")

        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        predictions = self.model.predict(X_test)
        predictions_decoded = self.decode_failure_mode(predictions)

        prediction_df = pd.DataFrame({
            'item_id': X_test.index,
            'Failure mode (rf)': predictions_decoded
        })
        return prediction_df

    def save_predictions(self, output_path, predictions, step):
        file_path = f"{output_path}/rf_predictions_{step}.csv"
        predictions.to_csv(file_path, index=False)

    def display_results(self, df):
        st.dataframe(df)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Failure Modes by Time")
            st.bar_chart(df[['time (months)', 'Failure mode (rf)']].groupby('time (months)').size())
        with col2:
            st.write("Failure Modes by Source")
            st.bar_chart(df[['source', 'Failure mode (rf)']].groupby('source').size())

    def run_full_pipeline(self, train_df, pseudo_test_with_truth_df, test_df):
        """Run the full pipeline from training to saving and displaying results."""

        model_name = 'RandomForestClassifierModel'
        st.markdown(f"## {model_name}")
        output_path = 'data/output/submission/random_forest'
        # Step 1: Prepare data and train the model
        X_train, y_train = self.prepare_data(train_df)
        self.train(X_train, y_train)

        # Step 2: Cross-validation predictions and scoring
        X_cv, y_cv = self.prepare_data(pseudo_test_with_truth_df)
        predictions_cv = self.predict(X_cv)
        self.save_predictions(output_path, predictions_cv, step='cross-val')
        generate_submission_file(model_name, output_path, step='cross-val')
        score = calculate_score(output_path, step='cross-val')
        st.write(f"Score de cross validation pour {model_name}: {score}")

        # Step 3: Final test predictions and scoring
        X_test, _ = self.prepare_data(test_df, target_col=None)
        predictions_test = self.predict(X_test)
        self.save_predictions(output_path, predictions_test, step='final-test')
        generate_submission_file(model_name, output_path, step='final-test')
        final_score = calculate_score(output_path, step='final-test')
        st.write(f"Le score final pour {model_name} est de {final_score}")
        st.dataframe(predictions_test)
