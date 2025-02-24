import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

class AnomalyDetection:
    def __init__(self):
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        self.one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)

    def detect_anomalies(self, X):
        iso_forest_anomalies = self.iso_forest.fit_predict(X)
        lof_anomalies = self.lof.fit_predict(X)
        one_class_svm_anomalies = self.one_class_svm.fit_predict(X)
        return iso_forest_anomalies, lof_anomalies, one_class_svm_anomalies

    def plot_anomalies(self, X, anomalies, title, feature_indices):
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(X, columns=[f'Feature {i}' for i in range(X.shape[1])])
        df['anomalies'] = anomalies

        plt.scatter(df.iloc[:, feature_indices[0]], df.iloc[:, feature_indices[1]], c='white', s=20, edgecolor='k')

        normal_points = df[df['anomalies'] == 1]
        anomaly_points = df[df['anomalies'] == -1]

        if not normal_points.empty:
            plt.scatter(normal_points.iloc[:, feature_indices[0]], normal_points.iloc[:, feature_indices[1]], c='blue', s=20, edgecolor='k', label="Нормальные точки")

        if not anomaly_points.empty:
            plt.scatter(anomaly_points.iloc[:, feature_indices[0]], anomaly_points.iloc[:, feature_indices[1]], c='red', s=20, edgecolor='k', label="Аномалии")

        plt.title(title)
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.legend()
        plt.show()