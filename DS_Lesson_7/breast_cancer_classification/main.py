from data_loader import DataLoader
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных
data_loader = DataLoader()
X, y = data_loader.get_features_and_target()

# Шаг 2: Разделение данных на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Обучение моделей
model_trainer = ModelTrainer()
fitted_models = model_trainer.train_models(X_train, y_train)

# Шаг 4: Оценка моделей
model_evaluator = ModelEvaluator(fitted_models, X_test, y_test)
evaluation_results = model_evaluator.evaluate_models()
print(evaluation_results)

# Шаг 5: Выбор лучшей модели
best_model = model_evaluator.get_best_model()
print(f"Лучший модель:\n{best_model}")