from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Gradient Boosting': GradientBoostingClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0),
            'AdaBoost': AdaBoostClassifier(),
            'Extra Trees': ExtraTreesClassifier(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'LightGBM': LGBMClassifier(),
            'K Neighbors': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'Dummy Classifier': DummyClassifier(strategy='most_frequent'),
            'SVM': SVC(kernel='linear', probability=True)
        }
        self.fitted_models = {}

    def train_models(self, X, y):
        for model_name, model in self.models.items():
            model.fit(X, y)
            self.fitted_models[model_name] = model
        return self.fitted_models