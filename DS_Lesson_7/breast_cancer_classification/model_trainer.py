import time
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
#import os
#import sys

# Список для сбора всех предупреждений и ошибок
errors_warnings = []

#Функция для добавления сообщений об ошибках и предупреждениях в список.
def log_message(message):
    errors_warnings.append(message)
    print(message)

class ModelTrainer:
    def __init__(self):
        #Инициализация классификаторов и их гиперпараметров.
        self.models = {
            'Gradient Boosting': GradientBoostingClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0),
            'AdaBoost': AdaBoostClassifier(),
            'Extra Trees': ExtraTreesClassifier(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'LightGBM': LGBMClassifier(),
            'K Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            #'XGBoost': XGBClassifier(eval_metric='logloss'),
            'Dummy Classifier': DummyClassifier(strategy='most_frequent'),
            'SVM': SVC(kernel='linear', probability=True)
        }
        self.param_grids = {
            'Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200]},
            'CatBoost': {'depth': [4, 6, 8], 'iterations': [100, 200]},
            'AdaBoost': {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [50, 100]},
            'Extra Trees': {'n_estimators': [100, 200], 'max_features': ['sqrt', 'log2']},
            'Quadratic Discriminant Analysis': {},
            'LightGBM': {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]},
            'K Neighbors': {'n_neighbors': [3, 5, 7]},
            'Decision Tree': {'max_depth': [None, 10, 20]},
            #'XGBoost': {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]},
            'Dummy Classifier': {},
            'SVM': {'C': [0.1, 1, 10]}
        }
        self.best_params = {}
        self.fitted_models = {}
        self.training_times = {}

    #Обучение одной модели с использованием GridSearchCV и измерение времени.
    def train_single_model(self, model_name, model, X, y):
        try:
            print(f"Начало обучения {model_name}...")

            start_time = time.time()
            param_grid = self.param_grids.get(model_name, {})
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
            grid_search.fit(X, y)
            end_time = time.time()
            training_time = end_time - start_time

            self.best_params[model_name] = grid_search.best_params_
            self.fitted_models[model_name] = grid_search.best_estimator_
            self.training_times[model_name] = training_time
            print(f"Обучение {model_name} завершено. Время: {training_time:.2f} секунд.")

        except Exception as e:
            log_message(f"Ошибка при обучении модели {model_name}: {e}")

    #Обучение всех моделей последовательно.
    def train_models(self, X, y):
        for model_name, model in self.models.items():
            self.train_single_model(model_name, model, X, y)
        return self.fitted_models

        """
        Обучение всех моделей в параллельных потоках.
        num_cores = multiprocessing.cpu_count()
        print(f"Количество доступных ядер: {num_cores}")
        Parallel(n_jobs=num_cores)(delayed(self.train_single_model)(model_name, model, X, y)
        for model_name, model in self.models.items())
        print(f"Количество потоков: {num_cores}")
        return self.fitted_models
        """

    #Возвращает лучшие гиперпараметры для каждой модели.
    def get_best_params(self):
        return self.best_params

    #Возвращает время, затраченное на обучение каждой модели.
    def get_training_times(self):
        return self.training_times    