from data_preprocessing import DataPreprocessing
from classification import Classification
from clustering import Clustering
from anomaly_detection import AnomalyDetection

def main():
    # Загрузка и предобработка данных
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    data_preprocessing = DataPreprocessing(url, columns)
    data = data_preprocessing.load_data()
    X_train, X_test, y_train, y_test = data_preprocessing.preprocess_data(data)
    
    # Классификация
    classification = Classification()
    best_estimators = classification.train_and_evaluate(X_train, X_test, y_train, y_test)
    classification.voting_classifier(best_estimators, X_train, X_test, y_train, y_test)
    
    # Сегментация данных
    clustering = Clustering()
    clusters = clustering.segment_data(data)
    clustering.plot_clusters(data, clusters)
    
    # Обнаружение аномалий
    anomaly_detection = AnomalyDetection()
    iso_forest_anomalies, lof_anomalies, one_class_svm_anomalies = anomaly_detection.detect_anomalies(X_train)
    
    # Визуализация аномалий
    anomaly_detection.plot_anomalies(X_train, iso_forest_anomalies, 'Аномалии Isolation Forest', [0, 1])
    anomaly_detection.plot_anomalies(X_train, lof_anomalies, 'Аномалии Local Outlier Factor', [0, 1])
    anomaly_detection.plot_anomalies(X_train, one_class_svm_anomalies, 'Аномалии OneClass SVM', [0, 1])

if __name__ == "__main__":
    main()