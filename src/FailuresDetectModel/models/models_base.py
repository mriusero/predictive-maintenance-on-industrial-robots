# models_base.py
class ModelBase:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def save_predictions(self, df, output_path):
        raise NotImplementedError

    def display_results(self, df):
        raise NotImplementedError



