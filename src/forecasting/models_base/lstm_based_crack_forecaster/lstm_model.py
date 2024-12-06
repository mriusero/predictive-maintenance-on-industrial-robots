import numpy as np
import pandas as pd
import streamlit as st
from src.forecasting.models_base.models_base import ModelBase

from src.forecasting.preprocessing.features import FeatureAdder
from src.forecasting.validation.display import DisplayData


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Masking, Input, Dropout

class LSTMModel(ModelBase):
    def __init__(self, min_sequence_length, forecast_months):
        super().__init__()
        self.min_sequence_length = min_sequence_length
        self.forecast_months = forecast_months
        self.model = None  # Model initialisation to None

    def train(self, X_train, y_train):
        # Print the shape and type of training data
        print("Training input shape:", X_train.shape)
        print("Training input type:", type(X_train))
        print("Training target shapes:")
        print("lengths_filtered_output:", y_train['lengths_filtered_output'].shape)
        print("lengths_measured_output:", y_train['lengths_measured_output'].shape)

        # Correct input shape based on data dimensions
        input_shape = (self.min_sequence_length, X_train.shape[2])

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
        print(np.isnan(X_train).any(), np.isinf(X_train).any())
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
                    X_train,
                    y_train_reshaped,
                    epochs=1,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=1
                )
                progress_bar.progress((epoch + 1) / num_epochs)
            progress_bar.empty()

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        with st.spinner('Generating predictions...'):
            progress_bar = st.progress(0)
            num_batches = len(X_test) // 32 + (len(X_test) % 32 != 0)
            predictions = []

            for i in range(num_batches):
                batch = X_test[i * 32:(i + 1) * 32]
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

    def prepare_train_sequences(self, df):
        item_indices = df['item_id'].unique()
        sequences = []
        targets_filtered = []
        targets_measured = []

        for item_index in item_indices:
            item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')

            # Print the columns being used for sequence preparation
            print(f"Preparing training sequences for item_id: {item_index}")
            print("Columns used:", item_data.columns.tolist())

            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values
            rolling_means_filtered = item_data['rolling_mean_length_filtered'].values
            rolling_stds_filtered = item_data['rolling_std_length_filtered'].values
            rolling_maxs_filtered = item_data['rolling_max_length_filtered'].values
            rolling_mins_filtered = item_data['rolling_min_length_filtered'].values
            #rolling_means_measured = item_data['rolling_mean_length_measured'].values
            #rolling_stds_measured = item_data['rolling_std_length_measured'].values
            #rolling_maxs_measured = item_data['rolling_max_length_measured'].values
            #rolling_mins_measured = item_data['rolling_min_length_measured'].values

            print(f"item_id: {item_index}, Length of data: {len(times)}")

            sequence_length = self.min_sequence_length

            for i in range(len(times) - sequence_length - self.forecast_months + 1):
                seq = np.column_stack((
                    times[i:i + sequence_length],
                    lengths_filtered[i:i + sequence_length],
                    lengths_measured[i:i + sequence_length],
                    rolling_means_filtered[i:i + sequence_length],
                    rolling_stds_filtered[i:i + sequence_length],
                    rolling_maxs_filtered[i:i + sequence_length],
                    rolling_mins_filtered[i:i + sequence_length],
             #      rolling_means_measured[i:i + sequence_length],
             #      rolling_stds_measured[i:i + sequence_length],
             #      rolling_maxs_measured[i:i + sequence_length],
             #      rolling_mins_measured[i:i + sequence_length]
                ))
                sequences.append(seq)

                target_filtered = lengths_filtered[i + sequence_length:i + sequence_length + self.forecast_months]
                target_measured = lengths_measured[i + sequence_length:i + sequence_length + self.forecast_months]

                targets_filtered.append(target_filtered)
                targets_measured.append(target_measured)

        if len(sequences) == 0:
            raise ValueError("No valid sequence was created with the provided data.")

        sequences_padded = np.array(
            pad_sequences(sequences, maxlen=self.min_sequence_length, padding='post', dtype='float32'))

        targets_filtered = np.array(targets_filtered).reshape(-1, self.forecast_months)
        targets_measured = np.array(targets_measured).reshape(-1, self.forecast_months)

        return sequences_padded, {'lengths_filtered_output': targets_filtered,
                                  'lengths_measured_output': targets_measured}

    def predict_futures_values(self, df):
        if self.model is None:
            raise ValueError("The model has not been trained.")

        def extract_features(item_data):
            # Print the columns being used for predictions
            print("Extracting features for predictions.")
            print("Columns availables:", item_data.columns.tolist())

            features = {
                'times': item_data['time (months)'].values,
                'length_filtered': item_data['length_filtered'].values,
                'length_measured': item_data['length_measured'].values,
                'rolling_means_filtered': item_data['rolling_mean_length_filtered'].values,
                'rolling_stds_filtered': item_data['rolling_std_length_filtered'].values,
                'rolling_maxs_filtered': item_data['rolling_max_length_filtered'].values,
                'rolling_mins_filtered': item_data['rolling_min_length_filtered'].values,
            #    'rolling_means_measured': item_data['rolling_mean_length_measured'].values,
            #    'rolling_stds_measured': item_data['rolling_std_length_measured'].values,
            #    'rolling_maxs_measured': item_data['rolling_max_length_measured'].values,
            #    'rolling_mins_measured': item_data['rolling_min_length_measured'].values
            }
            return features

        def prepare_test_sequence(features):
            last_sequence = np.column_stack((
                features['times'][-self.min_sequence_length:],
                features['length_filtered'][-self.min_sequence_length:],
                features['length_measured'][-self.min_sequence_length:],
                features['rolling_means_filtered'][-self.min_sequence_length:],
                features['rolling_stds_filtered'][-self.min_sequence_length:],
                features['rolling_maxs_filtered'][-self.min_sequence_length:],
                features['rolling_mins_filtered'][-self.min_sequence_length:],
             #   features['rolling_means_measured'][-self.min_sequence_length:],
             #   features['rolling_stds_measured'][-self.min_sequence_length:],
             #   features['rolling_maxs_measured'][-self.min_sequence_length:],
             #   features['rolling_mins_measured'][-self.min_sequence_length:]
            ))
            return pad_sequences([last_sequence], maxlen=self.min_sequence_length, padding='post', dtype='float32')

        item_indices = df['item_id'].unique()
        all_predictions = []

        with st.spinner('Calculating future values...'):

            progress_bar = st.progress(0)
            num_items = len(item_indices)

            for idx, item_index in enumerate(item_indices):
                item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')

                features = extract_features(item_data)

                print("Columns used:", list(features.keys()))

                last_sequence_padded = prepare_test_sequence(features)
                # Assurez-vous que `self.model.predict` retourne bien les deux sorties
                pred_lengths_filtered, pred_lengths_measured = self.model.predict(last_sequence_padded)

                print("Prediction shapes: lengths_filtered:", pred_lengths_filtered.shape, "lengths_measured:",
                      pred_lengths_measured.shape)  # Vérifiez la forme des prédictions

                # Combinez les prédictions de manière appropriée
                combined_predictions = np.column_stack(
                    (pred_lengths_filtered.flatten(), pred_lengths_measured.flatten()))
                all_predictions.append(combined_predictions)

                progress_bar.progress((idx + 1) / num_items)

            progress_bar.empty()

        st.success('Future values calculation completed!')
        return all_predictions

    def add_predictions_to_data(self, df, predictions):
        def prepare_initial_data(item_data, item_index, source):
            times = item_data['time (months)'].values
            lengths_filtered = item_data['length_filtered'].values
            lengths_measured = item_data['length_measured'].values

            features = {
                'rolling_means_filtered': item_data['rolling_mean_length_filtered'].values,
                'rolling_stds_filtered': item_data['rolling_std_length_filtered'].values,
                'rolling_maxs_filtered': item_data['rolling_max_length_filtered'].values,
                'rolling_mins_filtered': item_data['rolling_min_length_filtered'].values,
              #  'rolling_means_measured': item_data['rolling_mean_length_measured'].values,
              #  'rolling_stds_measured': item_data['rolling_std_length_measured'].values,
              #  'rolling_maxs_measured': item_data['rolling_max_length_measured'].values,
              #  'rolling_mins_measured': item_data['rolling_min_length_measured'].values
            }

            data_dict = {
                'item_id': item_index,
                'time (months)': times,
                'length_filtered': lengths_filtered,
                'length_measured': lengths_measured,
                'source': source
            }
            data_dict.update(features)

            #if scenario[:] == 'Scenario2':
            #    data_dict.update({
            #        'label': item_data['label'].values,
            #        'true_rul': item_data['true_rul'].values
            #    })
            return pd.DataFrame(data_dict)

        item_indices = df['item_id'].unique()
        extended_data = []

        for idx, item_index in enumerate(item_indices):
            item_data = df[df['item_id'] == item_index].sort_values(by='time (months)')
            max_time = np.max(item_data['time (months)'].values)
            forecast_length = len(predictions[idx])
            future_times = np.arange(np.ceil(max_time + 1), np.ceil(max_time + 1) + forecast_length)

            future_lengths_filtered = predictions[idx][:, 0]
            future_lengths_measured = predictions[idx][:, 1]

            initial_data = prepare_initial_data(item_data, item_index, source=0)
            forecast_data = pd.DataFrame({
                'item_id': item_index,
                'time (months)': future_times,
                'length_filtered': future_lengths_filtered,
                'length_measured': future_lengths_measured,
                'source': 1
            })
            extended_data.append(pd.concat([initial_data, forecast_data]))

        if not extended_data:
            raise ValueError("No extended data was created with the provided predictions.")

        df_extended = pd.concat(extended_data).reset_index(drop=True)
        df_extended['crack_failure'] = (
                    (df_extended['length_measured'] >= 0.85) | (df_extended['length_filtered'] >= 0.85)).astype(int)

        feature_adder = FeatureAdder(min_sequence_length=self.min_sequence_length)
        df_extended = feature_adder.add_features(df_extended, particles_filtery=False)

        #df_extended['item_id'] = df_extended['item_index'].astype(str)
        #df_extended.loc[:, 'item_index'] = df_extended['item_index'].apply(lambda x: f'item_{x}')

        return df_extended

    def save_predictions(self, output_path, df, step):

        file_path = f"{output_path}/lstm_predictions_{step}.csv"
        df.to_csv(file_path, index=False)

        return f"Predictions saved successfully: {output_path}"

    def display_results(self, df):
        display = DisplayData(df)

        col1, col2 = st.columns(2)
        with col1:
            display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'source')
            display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'item_id')
            display.plot_discrete_scatter(df, 'time (months)', 'length_measured', 'crack_failure')
        with col2:
            display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'source')
            display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'item_id')
            display.plot_discrete_scatter(df, 'time (months)', 'length_filtered', 'crack_failure')

        # Remove the line that plots based on Failure mode as it's not used in V4
        # plot_scatter2(df, 'time (months)', 'length_measured', 'Failure mode (lstm)')

    def run_full_pipeline(self, train_df, pseudo_test_with_truth_df, test_df):
        """
        Méthode pour exécuter le pipeline complet: entraînement, prédictions, et génération des fichiers de soumission.

        Args:
            train_df (DataFrame): Les données d'entraînement.
            pseudo_test_with_truth_df (DataFrame): Données de test avec vérités terrain pour la validation croisée.
            test_df (DataFrame): Les données de test finales.
            output_path (str): Le chemin de sauvegarde des résultats.
            model_name (str): Le nom du modèle.
        """
        model_name = 'LSTMModel'
        st.markdown(f"## {model_name}")
        output_path = 'data/output/submission/lstm'
        # Préparation des séquences de formation
        X_, y_ = self.prepare_train_sequences(train_df)

        # Entraînement du modèle
        self.train(X_, y_)

        # Validation croisée sur pseudo_test_with_truth_df
        all_predictions = self.predict_futures_values(pseudo_test_with_truth_df)
        lstm_predictions_cross_val = self.add_predictions_to_data(pseudo_test_with_truth_df, all_predictions)
        self.save_predictions(output_path, lstm_predictions_cross_val, step='cross-val')

        # Génération du fichier de soumission et calcul du score pour la validation croisée
        #generate_submission_file(model_name, output_path, step='cross-val')
        #score = calculate_score(output_path, step='cross-val')
        #st.write(f"Score de cross validation pour {model_name}: {score}")

        # Prédictions finales sur test_df
        all_predictions = self.predict_futures_values(test_df)
        lstm_predictions_final_test = self.add_predictions_to_data(test_df, all_predictions)
        self.save_predictions(output_path, lstm_predictions_final_test, step='final-test')

        # Génération du fichier de soumission et calcul du score final
        #generate_submission_file(model_name, output_path, step='final-test')
        #final_score = calculate_score(output_path, step='final-test')
        #st.write(f"Le score final pour {model_name} est de {final_score}")

        # Affichage des résultats
        self.display_results(lstm_predictions_final_test)

        return lstm_predictions_cross_val, lstm_predictions_final_test