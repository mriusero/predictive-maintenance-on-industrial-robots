from src.core.generate_data import generate_pseudo_testing_data, generate_pseudo_testing_data_with_truth


def fleet_management_pipeline(train_df, pseudo_test_with_truth_df, test_df):
    """
    Pipeline function that runs the fleet management.
    """

    generate_pseudo_testing_data('data/input/training_data/pseudo_testing_data_with_truth',
                                 'data/input/training_data/degradation_data')

    generate_pseudo_testing_data_with_truth('data/input/training_data/pseudo_testing_data_with_truth',
                                            'data/input/training_data/pseudo_testing_data')