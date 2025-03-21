import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self):
        self.data = None
        self.target = None

    def load_data(self):
        """
        Загружает датасет California Housing.
        """
        housing = fetch_california_housing()
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.target = housing.target

    def split_data(self, test_size=0.2, random_state=42):
        """
        Разделяет данные на обучающие и тестовые выборки.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        """
        Масштабирует данные.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled