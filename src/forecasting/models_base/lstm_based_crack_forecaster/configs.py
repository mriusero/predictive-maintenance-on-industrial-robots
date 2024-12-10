MODEL_NAME = 'lstm_based_crack_forecaster'
MODEL_FOLDER = 'models/' + MODEL_NAME + '/'
MODEL_PATH = MODEL_FOLDER + MODEL_NAME + '_model.pkl'
HYPERPARAMETERS_PATH = MODEL_FOLDER + MODEL_NAME + '_best_params.pkl'
LOG_PATH = MODEL_FOLDER + MODEL_NAME + '_study_logs.json'
MIN_SEQUENCE_LENGTH = 2
FORECAST_MONTHS = 6
SUBMISSION_FOLDER = 'data/output/submission/lstm_based_crack_forecaster'

FEATURE_COLUMNS = [
    'time (months)',

    'beta0', 'beta1', 'beta2',

    'length_filtered', 'length_measured',
    'Infant mortality', 'Control board failure', 'Fatigue crack',

    'rolling_mean_time (months)', 'static_mean_time (months)',
    'rolling_std_time (months)', 'static_std_time (months)',
    'rolling_max_time (months)', 'static_max_time (months)',
    'rolling_min_time (months)', 'static_min_time (months)',

    'rolling_mean_length_filtered', 'static_mean_length_filtered',
    'rolling_std_length_filtered', 'static_std_length_filtered',
    'rolling_max_length_filtered', 'static_max_length_filtered',
    'rolling_min_length_filtered', 'static_min_length_filtered',

    'length_filtered_shift_1', 'length_filtered_shift_2', 'length_filtered_shift_3',
    'length_filtered_shift_4', 'length_filtered_shift_5', 'length_filtered_shift_6',

    'length_filtered_ratio_1-2', 'length_filtered_ratio_2-3', 'length_filtered_ratio_3-4',
    'length_filtered_ratio_4-5', 'length_filtered_ratio_5-6',

    'Trend', 'Seasonal', 'Residual'
]

TARGET_COLUMNS = [
    'length_filtered', 'length_measured',                           # Forecasted lengths (6 months)
    'Infant mortality', 'Control board failure', 'Fatigue crack'    # Classification targets
]