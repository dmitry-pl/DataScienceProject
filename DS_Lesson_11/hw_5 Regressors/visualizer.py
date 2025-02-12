import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    def __init__(self, results, y_preds, y_test):
        self.results = results
        self.y_preds = y_preds
        self.y_test = y_test

    # Визуализирует метрики всех моделей.
    def plot_metrics(self):
        results_df = pd.DataFrame(self.results).T
        plt.figure(figsize=(18, 12))

        # RMSE
        plt.subplot(3, 3, 1)
        sns.barplot(x=results_df.index, y='RMSE', data=results_df)
        plt.title('RMSE of Different Regressors')
        plt.xticks(rotation=45)

        # R2
        plt.subplot(3, 3, 2)
        sns.barplot(x=results_df.index, y='R2', data=results_df)
        plt.title('R2 Score of Different Regressors')
        plt.xticks(rotation=45)

        # MAPE
        plt.subplot(3, 3, 3)
        sns.barplot(x=results_df.index, y='MAPE', data=results_df)
        plt.title('MAPE of Different Regressors')
        plt.xticks(rotation=45)

        # MedAE
        plt.subplot(3, 3, 4)
        sns.barplot(x=results_df.index, y='MedAE', data=results_df)
        plt.title('MedAE of Different Regressors')
        plt.xticks(rotation=45)

        # Время выполнения
        plt.subplot(3, 3, 5)
        sns.barplot(x=results_df.index, y='Time', data=results_df)
        plt.title('Training Time of Different Regressors')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    # Визуализирует истинные и предсказанные значения для каждой модели.
    def plot_predictions(self):
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