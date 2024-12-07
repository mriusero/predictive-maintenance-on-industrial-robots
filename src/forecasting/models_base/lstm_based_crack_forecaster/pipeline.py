import streamlit as st
import datetime

from .configs import MODEL_NAME, MODEL_PATH, SUBMISSION_FOLDER, MIN_SEQUENCE_LENGTH, FORECAST_MONTHS
from .evaluation import save_predictions, display_results
from .forecasting import predict_futures_values
from .helper import add_predictions_to_data
from .model import LSTMModel
from .processing import prepare_train_sequences
from ...validation.validation import generate_submission_file, calculate_score


def lstm_training_pipeline(train_df, pseudo_test_with_truth_df, optimize=False):
    """
    Function that runs the training pipeline for the LSTM crack growth forecast model.
    """
    # Prepare the training sequences
    x_, y_ = prepare_train_sequences(train_df, min_sequence_length=MIN_SEQUENCE_LENGTH, forecast_months=FORECAST_MONTHS)

    # Instantiate the LSTM model
    model = LSTMModel(min_sequence_length=MIN_SEQUENCE_LENGTH, forecast_months=FORECAST_MONTHS)

    # Train the model
    model.train(x_, y_)

    # Validation
    all_predictions = predict_futures_values(model, pseudo_test_with_truth_df)

    # Submission
    lstm_predictions_cross_val = add_predictions_to_data(pseudo_test_with_truth_df, all_predictions, min_sequence_length=MIN_SEQUENCE_LENGTH)
    save_predictions(SUBMISSION_FOLDER, lstm_predictions_cross_val, step='cross-val')
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')

    # Calculate the score
    score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')
    st.write(f"Score de cross validation pour {MODEL_NAME}: {score}")

    # Save Model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_with_timestamp = f"{MODEL_PATH.rsplit('.', 1)[0]}_{timestamp}.pkl"
    try:
        model.save_model(path=model_path_with_timestamp)
        st.success(f"Model saved successfully at {model_path_with_timestamp}")
    except Exception as e:
        st.error(f"Failed to save model: {e}")


def lstm_testing_pipeline(train_df, test_df, optimize=False):
    """
    Function that runs the testing pipeline for the LSTM crack growth forecast model.
    """
    # Model loading
    model = LSTMModel.load_model(path=MODEL_PATH)

    # Prédictions finales sur test_df
    all_predictions = predict_futures_values(model, test_df)
    lstm_predictions_final_test = add_predictions_to_data(test_df, all_predictions, min_sequence_length=MIN_SEQUENCE_LENGTH)
    save_predictions(SUBMISSION_FOLDER, lstm_predictions_final_test, step='final-test')

    # Génération du fichier de soumission et calcul du score final
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='final-test')
    final_score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='final-test')
    st.write(f"Le score final pour {MODEL_NAME} est de {final_score}")

    # Affichage des résultats
    display_results(lstm_predictions_final_test)






