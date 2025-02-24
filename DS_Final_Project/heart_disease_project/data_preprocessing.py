import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, url, columns):
        self.url = url
        self.columns = columns

    def load_data(self):
        data = pd.read_csv(self.url, names=self.columns, na_values='?')
        data.fillna(data.median(), inplace=True)
        return data

    def preprocess_data(self, data):
        X = data.drop('num', axis=1)
        y = data['num']

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

        poly = PolynomialFeatures(degree=2, interaction_only=True)
        X_poly = poly.fit_transform(X)

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test