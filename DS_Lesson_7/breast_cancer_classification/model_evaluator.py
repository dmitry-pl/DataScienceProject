from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

class ModelEvaluator:
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

    def evaluate_models(self):
        results_list = []
        for model_name, model in self.models.items():
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
        self.results = pd.concat([self.results, pd.DataFrame(results_list)], ignore_index=True)
        return self.results

    def get_best_model(self):
        best_model = self.results.loc[self.results['ROC AUC'].idxmax()]
        return best_model