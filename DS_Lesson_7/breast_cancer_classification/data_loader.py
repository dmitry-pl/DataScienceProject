import pandas as pd
from sklearn.datasets import load_breast_cancer

class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self):
        # Загружаем набор данных Breast Cancer Wisconsin
        cancer_data = load_breast_cancer()
        self.data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
        self.data['target'] = cancer_data.target
        return self.data

    def get_features_and_target(self):
        if self.data is None:
            self.load_data()
        X = self.data.drop(columns=['target'])
        y = self.data['target']
        return X, y