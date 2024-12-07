import streamlit as st

from .processing import prepare_train_sequences
from .model import LSTMModel
from .forecasting import predict_futures_values
from .helper import add_predictions_to_data
from .evaluation import save_predictions, display_results

from ...validation.validation import generate_submission_file, calculate_score

from .configs import MODEL_NAME, SUBMISSION_FOLDER, MIN_SEQUENCE_LENGTH, FORECAST_MONTHS


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

    # Validation croisée sur pseudo_test_with_truth_df
    all_predictions = predict_futures_values(model, pseudo_test_with_truth_df)
    lstm_predictions_cross_val = add_predictions_to_data(pseudo_test_with_truth_df, all_predictions, min_sequence_length=MIN_SEQUENCE_LENGTH)
    save_predictions(SUBMISSION_FOLDER, lstm_predictions_cross_val, step='cross-val')

    # Génération du fichier de soumission et calcul du score pour la validation croisée
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')
    score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')
    st.write(f"Score de cross validation pour {MODEL_NAME}: {score}")


#def lstm_testing_pipeline(train_df, test_df, optimize=False):
#
#    # Prédictions finales sur test_df
#    all_predictions = predict_futures_values(model, test_df)
#    lstm_predictions_final_test = self.add_predictions_to_data(test_df, all_predictions)
#    self.save_predictions(output_path, lstm_predictions_final_test, step='final-test')
#
#    # Génération du fichier de soumission et calcul du score final
#    #generate_submission_file(model_name, output_path, step='final-test')
#    #final_score = calculate_score(output_path, step='final-test')
#    #st.write(f"Le score final pour {model_name} est de {final_score}")
#
#    # Affichage des résultats
#    self.display_results(lstm_predictions_final_test)
#    return lstm_predictions_cross_val, lstm_predictions_final_test
#
#    # Instantiate the LSTM model
#    #model = LSTMModel(min_sequence_length=2, forecast_months=6)
#
#    #odel.run_full_pipeline(train_df, pseudo_test_with_truth_df, optimize=optimize)
#
#    #return lstm_model


