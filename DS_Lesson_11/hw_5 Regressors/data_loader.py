import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self):
        self.data = None
        self.target = None

    # Загружает датасет California Housing и разделяет его на обучающую и тестовую выборки.
    def load_data(self):
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data['PRICE'] = housing.target
        self.data = data.drop('PRICE', axis=1)
        self.target = data['PRICE']

    # Разделяет данные на обучающие и тестовые выборки.
    def split_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    # Масштабирует данные.
    def scale_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled