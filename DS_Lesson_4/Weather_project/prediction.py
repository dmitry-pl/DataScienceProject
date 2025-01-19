import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class Prediction:
    def __init__(self, data, target_column):  # Инициализация с передачей DataFrame и столбца целевой переменной
        self.data = data
        self.target_column = target_column

    def preprocess_data(self):
        # Предобработка данных: отделение признаков и целевой переменной
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        return X, y

    def train_model(self, X_train, y_train):
        # Обучение модели линейной регрессии
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test):
        # Предсказание значений на тестовом наборе данных
        return model.predict(X_test)

    def plot_predictions(self, y_test, y_pred):
        # Визуализация истинных и предсказанных значений на примере 50-ти элементов
        sample_size = 50
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.index[:sample_size], y_test[:sample_size], label='Истинные значения')
        plt.plot(y_test.index[:sample_size], y_pred[:sample_size], label='Предсказанные значения', linestyle='--')
        plt.xlabel('Индекс')  # Обновление метки по оси X
        plt.ylabel('Температура')
        plt.title('Истинные и предсказанные значения температуры (пример 50-ти)')
        plt.legend()
        plt.show()