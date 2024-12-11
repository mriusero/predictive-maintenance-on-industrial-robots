from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .callbacks import TrainingPlot
from .configs import MODEL_PATH, MODEL_FOLDER, HYPERPARAMETERS_PATH, LOG_PATH
from .evaluation import display_results
from .forecasting import predict_future_values
from .helper import add_predictions_to_data
from .model import LSTMModel
from .processing import prepare_sequences


def lstm_training_pipeline(train_df, optimize=False):
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
    earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    plot_losses = TrainingPlot(['length_filtered', 'length_measured', 'Infant_mortality', 'Control_board_failure', 'Fatigue_crack'])

    # Training
    lstm.train(
        x_, y_,
        epochs=100,
        batch_size=32,
        validation_split=0.3,
        callbacks=[plot_losses, earlystop, reduce_lr],
        optimize=optimize,
        n_trials=50,
        params_path=HYPERPARAMETERS_PATH,
        log_path=LOG_PATH
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
    Function that runs the validation pipeline for the LSTM crack growth forecast model.
    """
    # Load trained model
    lstm = LSTMModel()
    model = lstm.load_model(path=MODEL_PATH)

    # Forecasting & classification
    predict_future_values(
        model, test_df,
        output_file_path=f'{MODEL_FOLDER}'+'predictions.json'
    )

    # Submission
    df_extended = add_predictions_to_data(
        initial_data=test_df,
    )

    display_results(df_extended)

    # Calculate the score
    #save_predictions(output_path=SUBMISSION_FOLDER, df=df_extended, step='cross-val')
    #generate_submission_file(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')
    #score = calculate_score(model_name=MODEL_NAME, submission_path=SUBMISSION_FOLDER, step='cross-val')
    #st.write(f"Score de cross validation pour {MODEL_NAME}: {score}")






