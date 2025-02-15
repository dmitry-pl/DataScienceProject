import time
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error

class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.results = {}
        self.y_preds = {}
        self.best_params = {}

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """
        Обучает и оценивает модель, возвращает метрики и предсказания.
        """
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        elapsed_time = end_time - start_time
        return mae, mse, rmse, r2, mape, medae, elapsed_time, y_pred

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Обучает и оценивает все модели с использованием GridSearchCV и кросс-валидации.
        """
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for name, (model, params) in self.models.items():
            grid_search = GridSearchCV(model, params, cv=kfold, scoring='r2')
            mae, mse, rmse, r2, mape, medae, elapsed_time, y_pred = self.evaluate_model(grid_search, X_train, y_train, X_test, y_test)
            self.results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'MedAE': medae, 'Time': elapsed_time}
            self.y_preds[name] = y_pred
            self.best_params[name] = grid_search.best_params_
        return self.results, self.y_preds, self.best_params