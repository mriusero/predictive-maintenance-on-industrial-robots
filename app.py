from src.core import generate_pseudo_testing_data, generate_pseudo_testing_data_with_truth
from src.app import app_layout

def main():

    generate_pseudo_testing_data('data/input/training_data/pseudo_testing_data_with_truth',
                                 'data/input/training_data/degradation_data')

    generate_pseudo_testing_data_with_truth('data/input/training_data/pseudo_testing_data_with_truth',
                                            'data/input/training_data/pseudo_testing_data')

    app_layout()

if __name__ == '__main__':
    main()
