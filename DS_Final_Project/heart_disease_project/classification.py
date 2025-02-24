from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

class Classification:
    def __init__(self):
        self.models = {
            'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
            'Random Forest': (RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
            'Gradient Boosting': (GradientBoostingClassifier(), {'learning_rate': [0.1, 0.5], 'n_estimators': [100, 200]}),
            'Support Vector Machine': (SVC(), {'C': [1, 10], 'kernel': ['linear', 'rbf']}),
            'K-Neighbors Classifier': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
            'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {'learning_rate': [0.1, 0.5], 'n_estimators': [100, 200]}),
            'CatBoost': (CatBoostClassifier(verbose=0), {'iterations': [500, 1000], 'learning_rate': [0.1, 0.5]})
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        best_estimators = {}
        for name, (model, params) in self.models.items():
            print(f'Training {name}...')
            grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            end_time = time.time()
            best_estimators[name] = grid_search.best_estimator_
            print(f'{name} best parameters: {grid_search.best_params_}')
            print(f'{name} best score: {grid_search.best_score_:.4f}')
            print(f'{name} Время выполнения: {end_time - start_time:.2f} секунд')

        return best_estimators

    def voting_classifier(self, best_estimators, X_train, X_test, y_train, y_test):
        ensemble_estimators = [
            ('log_reg', best_estimators['Logistic Regression']),
            ('rf', best_estimators['Random Forest']),
            ('gb', best_estimators['Gradient Boosting']),
            ('svm', best_estimators['Support Vector Machine']),
            ('knn', best_estimators['K-Neighbors Classifier']),
            ('xgb', best_estimators['XGBoost']),
            ('cat', best_estimators['CatBoost'])
        ]

        ensemble_clf = VotingClassifier(estimators=ensemble_estimators, voting='hard')
        start_time = time.time()
        ensemble_clf.fit(X_train, y_train)
        ensemble_y_pred = ensemble_clf.predict(X_test)
        end_time = time.time()
        ensemble_accuracy = accuracy_score(y_test, ensemble_y_pred)
        print(f'Ensemble Model Accuracy: {ensemble_accuracy:.4f}')
        print(classification_report(y_test, ensemble_y_pred))

        conf_matrix = confusion_matrix(y_test, ensemble_y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
        plt.title('Confusion Matrix for Ensemble Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        print(f'Ensemble Model Время выполнения: {end_time - start_time:.2f} секунд')