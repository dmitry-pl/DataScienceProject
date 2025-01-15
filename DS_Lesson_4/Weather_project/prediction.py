### Файл `prediction.py`
#Этот модуль отвечает за предсказание температур и визуализацию результатов.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class Prediction:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column

    def preprocess_data(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return X, y

    def train_model(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test):
        return model.predict(X_test)

    def plot_predictions(self, y_test, y_pred):
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.index, y_test, label='Истинные значения')
        plt.plot(y_test.index, y_pred, label='Предсказанные значения', linestyle='--')
        plt.xlabel('Дата')
        plt.ylabel('Температура')
        plt.title('Истинные и предсказанные значения температуры')
        plt.legend()
        plt.show()