import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    def __init__(self, results, y_preds, y_test):
        self.results = results
        self.y_preds = y_preds
        self.y_test = y_test

    def plot_metrics(self):
        """
        Визуализирует метрики всех моделей.
        """
        results_df = pd.DataFrame(self.results).T
        n_metrics = len(results_df.columns)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Расчет количества строк
        plt.figure(figsize=(18, 6 * n_rows))

        for i, metric in enumerate(results_df.columns, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.barplot(x=results_df.index, y=metric, data=results_df)
            plt.title(f'{metric} of Different Regressors')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_predictions(self):
        """
        Визуализирует истинные и предсказанные значения для каждой модели.
        """
        n_models = len(self.y_preds)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols  # Расчет количества строк

        plt.figure(figsize=(18, 6 * n_rows))

        for i, (name, y_pred) in enumerate(self.y_preds.items(), 1):
            plt.subplot(n_rows, n_cols, i)
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], '--', color='red')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{name} Predictions')

        plt.tight_layout()
        plt.show()
