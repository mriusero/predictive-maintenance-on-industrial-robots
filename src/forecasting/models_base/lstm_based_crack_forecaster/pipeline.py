#pipeline.py

import streamlit as st

from .configs import MODEL_NAME, MODEL_PATH, MODEL_FOLDER, SUBMISSION_FOLDER, MIN_SEQUENCE_LENGTH
from .evaluation import save_predictions, display_results, TrainingPlot
from .forecasting import predict_future_values
from .helper import add_predictions_to_data
from .model import LSTMModel
from .processing import prepare_sequences
from ...validation.validation import generate_submission_file, calculate_score

from tensorflow.keras import callbacks

def lstm_training_pipeline(train_df, pseudo_test_with_truth_df, optimize=False):
    """
    Function that runs the training pipeline for the LSTM crack growth forecast model.
    """
    # Prepare the training sequences
    x_, y_ = prepare_sequences(
        data=train_df,
        mode='train',
    )

    # Instantiate the LSTM model
    lstm = LSTMModel()

    # Callbacks
    earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    output_names = ['length_filtered', 'length_measured', 'Infant_mortality', 'Control_board_failure', 'Fatigue_crack']
    plot_losses = TrainingPlot(output_names=output_names)

    # Training
    lstm.train(
        x_, y_,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[plot_losses, earlystop]
    )


def lstm_validation_pipeline(train_df, pseudo_test_with_truth_df, optimize=False):
    """
    Function that runs the validation pipeline for the LSTM crack growth forecast model.
    """
    # Load trained model
    lstm = LSTMModel()
    model = lstm.load_model(path=MODEL_PATH)

    # Forecasting & classification
    predict_future_values(
        model, pseudo_test_with_truth_df,
        output_file_path=f'{MODEL_FOLDER}'+'predictions.json'
    )

    # Submission
    df_extended = add_predictions_to_data(
        initial_data=pseudo_test_with_truth_df,
    )

    display_results(df_extended)

    # Calculate the score
    #save_predictions(output_path=SUBMISSION_FOLDER, df=df_extended, step='cross-val')
    #generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')
    #score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')
    #st.write(f"Score de cross validation pour {MODEL_NAME}: {score}")



def lstm_testing_pipeline(train_df, test_df, optimize=False):
    """
    Function that runs the testing pipeline for the LSTM crack growth forecast model.
    """
    # Model loading
    model = LSTMModel.load_model(path=MODEL_PATH)

    # Prédictions finales sur test_df
    all_predictions = predict_future_values(model, test_df)
    lstm_predictions_final_test = add_predictions_to_data(test_df, all_predictions, min_sequence_length=MIN_SEQUENCE_LENGTH)
    save_predictions(SUBMISSION_FOLDER, lstm_predictions_final_test, step='final-test')

    # Génération du fichier de soumission et calcul du score final
    generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='final-test')
    final_score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='final-test')
    st.write(f"Le score final pour {MODEL_NAME} est de {final_score}")

    # Affichage des résultats
    display_results(lstm_predictions_final_test)






