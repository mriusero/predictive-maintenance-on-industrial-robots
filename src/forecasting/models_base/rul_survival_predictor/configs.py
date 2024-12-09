MODEL_NAME = 'rul_survival_predictor'
MODEL_FOLDER = 'models/' + MODEL_NAME + '/'
MODEL_PATH = MODEL_FOLDER + MODEL_NAME + '_model.pkl'
HYPERPARAMETERS_PATH = MODEL_FOLDER + MODEL_NAME + '_best_params.pkl'
SUBMISSION_FOLDER = 'data/output/submission'

SELECTED_VARIABLES = ["source", "item_id", "time (months)", "label",                    # "length_measured",
                      "Infant mortality", "Control board failure", "Fatigue crack",     # "Failure mode"
                      "Time to failure (months)", "rul (months)", "failure_month",      # "true_rul",
                      "length_filtered", "beta0", "beta1", "beta2",

                      "rolling_mean_time (months)", "static_mean_time (months)",
                      "rolling_std_time (months)", "static_std_time (months)",
                      "rolling_max_time (months)", "static_max_time (months)",
                      "rolling_min_time (months)", "static_min_time (months)",

                      "rolling_mean_length_filtered", "static_mean_length_filtered",
                      "rolling_std_length_filtered", "static_std_length_filtered",
                      "rolling_max_length_filtered", "static_max_length_filtered",
                      "rolling_min_length_filtered", "static_min_length_filtered",
                      "length_filtered_shift_1",
                      "length_filtered_shift_2", "length_filtered_shift_3",
                      "length_filtered_shift_4", "length_filtered_shift_5",
                      "length_filtered_shift_6", "length_filtered_ratio_1-2",
                      "length_filtered_ratio_2-3", "length_filtered_ratio_3-4",
                      "length_filtered_ratio_4-5", "length_filtered_ratio_5-6",
                      "Trend", "Seasonal", "Residual"]

                    # "rolling_mean_length_measured", "static_mean_length_measured",
                    # "rolling_std_length_measured", "static_std_length_measured",
                    # "rolling_max_length_measured", "static_max_length_measured",
                    # "rolling_min_length_measured", "static_min_length_measured",
                    # "length_measured_shift_1", "length_measured_shift_2",
                    # "length_measured_shift_3", "length_measured_shift_4",
                    # "length_measured_shift_5", "length_measured_shift_6",
                    # "length_measured_ratio_1-2", "length_measured_ratio_2-3",
                    # "length_measured_ratio_3-4", "length_measured_ratio_4-5",
                    # "length_measured_ratio_5-6",