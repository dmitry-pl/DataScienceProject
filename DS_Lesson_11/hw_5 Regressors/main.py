import logging
from data_loader import DataLoader
from model_trainer import ModelTrainer
from visualizer import Visualizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Основная функция
def main():
    try:
        # Загрузка и предобработка данных
        data_loader = DataLoader()
        data_loader.load_data()
        X_train, X_test, y_train, y_test = data_loader.split_data()
        X_train, X_test = data_loader.scale_data(X_train, X_test)

        # Инициализация моделей и гиперпараметров
        regressors = {
            'Linear Regression': (LinearRegression(), {}),
            'Random Forest': (RandomForestRegressor(random_state=42), {'n_estimators': [50, 100, 200]}),
            'Support Vector Regressor': (SVR(), {'kernel': ['linear', 'rbf'], 'C': [1, 10]}),
            'K-Neighbors Regressor': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
            'Decision Tree': (DecisionTreeRegressor(random_state=42), {'max_depth': [None, 10, 20]})
        }

        # Обучение моделей и оценка
        model_trainer = ModelTrainer(regressors)
        results, y_preds, best_params = model_trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

        # Вывод всех метрик и показателей в консоль
        results_df = pd.DataFrame(results).T
        print(results_df)

        # Вывод лучших параметров
        print("Лучшие параметры для каждой модели:")
        for name, params in best_params.items():
            print(f"{name}: {params}")

        # Визуализация
        visualizer = Visualizer(results, y_preds, y_test)
        visualizer.plot_metrics()
        visualizer.plot_predictions()

        # Выбор лучшего регрессора
        best_model = results_df['R2'].idxmax()
        print(f'Лучший регрессор: {best_model}')

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()