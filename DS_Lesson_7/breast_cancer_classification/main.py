from data_loader import DataLoader
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
import warnings

# Список для сбора всех предупреждений и ошибок
errors_warnings = []

def log_message(message):
    """
    Функция для добавления сообщений об ошибках и предупреждениях в список.
    """
    errors_warnings.append(message)
    print(message)

# Игнорирование предупреждений и перенаправление их в log_message
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    log_message(f"Предупреждение: {message} в файле {filename}, строка {lineno}")

warnings.showwarning = custom_warning_handler

try:
    # Шаг 1: Загрузка данных
    data_loader = DataLoader()
    X, y = data_loader.get_features_and_target()
except Exception as e:
    log_message(f"Ошибка при загрузке данных: {e}")

try:
    # Шаг 2: Разделение данных на обучающие и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    log_message(f"Ошибка при разделении данных: {e}")

try:
    # Шаг 3: Обучение моделей
    model_trainer = ModelTrainer()
    fitted_models = model_trainer.train_models(X_train, y_train)
except Exception as e:
    log_message(f"Ошибка при обучении моделей: {e}")

try:
    # Шаг 4: Вывод времени обучения
    training_times = model_trainer.get_training_times()
    print("Время обучения для каждой модели:")
    for model_name, training_time in training_times.items():
        print(f"{model_name}: {training_time:.2f} секунд")
except Exception as e:
    log_message(f"Ошибка при выводе времени обучения: {e}")

try:
    # Шаг 5: Оценка моделей
    model_evaluator = ModelEvaluator(fitted_models, X_test, y_test)
    evaluation_results = model_evaluator.evaluate_models()
    print(evaluation_results)
except Exception as e:
    log_message(f"Ошибка при оценке моделей: {e}")

try:
    # Шаг 6: Вывод лучших гиперпараметров для каждого классификатора
    best_params = model_trainer.get_best_params()
    print(f"Лучшие гиперпараметры для каждого классификатора:\n{best_params}")
except Exception as e:
    log_message(f"Ошибка при выводе лучших гиперпараметров: {e}")

try:
    # Шаг 7: Выбор лучшей модели
    best_model = model_evaluator.get_best_model()
    print(f"Лучшая модель:\n{best_model}")
except Exception as e:
    log_message(f"Ошибка при выборе лучшей модели: {e}")

# Вывод всех предупреждений и ошибок в конце
if errors_warnings:
    print("\nСписок всех предупреждений и ошибок:")
    for message in errors_warnings:
        print(message)