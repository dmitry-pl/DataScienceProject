from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Список для сбора всех предупреждений и ошибок
errors_warnings = []

def log_message(message):
    """
    Функция для добавления сообщений об ошибках и предупреждениях в список.
    """
    errors_warnings.append(message)
    print(message)

class ModelEvaluator:
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

    def evaluate_models(self):
        """
        Оценивает модели по различным метрикам и строит матрицы истинности.
        """
        results_list = []
        for model_name, model in self.models.items():
            try:
                print(f"Начало оценки модели {model_name}...")
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
                results_list.append({
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'ROC AUC': roc_auc
                })
                self.plot_confusion_matrix(model_name, y_pred)
            except Exception as e:
                log_message(f"Ошибка при оценке модели {model_name}: {e}")
        self.results = pd.concat([self.results, pd.DataFrame(results_list)], ignore_index=True)
        return self.results

    def plot_confusion_matrix(self, model_name, y_pred):
        """
        Строит матрицу истинности для заданной модели и предсказанных значений.
        """
        try:
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f'Confusion Matrix for {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
        except Exception as e:
            log_message(f"Ошибка при построении матрицы истинности для модели {model_name}: {e}")

    def get_best_model(self):
        """
        Возвращает лучшую модель на основе метрики ROC AUC.
        """
        try:
            best_model = self.results.loc[self.results['ROC AUC'].idxmax()]
            return best_model
        except Exception as e:
            log_message(f"Ошибка при выборе лучшей модели: {e}")
            return None