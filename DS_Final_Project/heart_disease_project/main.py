import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка датасета
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'num'  # 'num' - целевая переменная (0 - отсутствие заболевания, 1 - наличие заболевания)
]
data = pd.read_csv(url, names=columns, na_values='?')

# Описание колонок
data.columns = [
    'age',        # Возраст
    'sex',        # Пол (1 - мужской, 0 - женский)
    'cp',         # Тип боли в груди (значения от 1 до 4)
    'trestbps',   # Артериальное давление в состоянии покоя
    'chol',       # Уровень холестерина
    'fbs',        # Уровень сахара в крови натощак (1 - больше 120 мг/дл, 0 - меньше)
    'restecg',    # Результаты электрокардиограммы (0, 1, 2)
    'thalach',    # Максимальная частота сердечных сокращений
    'exang',      # Наличие стенокардии при физической нагрузке (1 - да, 0 - нет)
    'oldpeak',    # Депрессия ST сегмента после нагрузки относительно покоя
    'slope',      # Наклон ST сегмента при максимальной нагрузке (1, 2, 3)
    'ca',         # Количество крупных сосудов (0-3)
    'thal',       # Талассемия (3 = нормальный, 6 = дефект фиксированный, 7 = дефект обратимый)
    'num'         # Целевая переменная (0 - отсутствие заболевания, 1 - наличие заболевания)
]

# Заполнение пропущенных значений медианой колонки
data.fillna(data.median(), inplace=True)

# Вывод первых 5 строк датасета
print(data.head())

# Добавление выводов информации о датасете
print(data.shape)
print(data.info())

# Выделение признаков и целевой переменной
X = data.drop('num', axis=1)
y = data['num']

# Разделение данных на обучающие и тестовые выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Вывод первых 5 строк преобразованных данных
print(pd.DataFrame(X_train, columns=data.columns[:-1]).head())


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Задание параметров для Grid Search
param_grid = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'K-Neighbors Classifier': {'n_neighbors': [3, 5, 7]}
}

# Инициализация моделей
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'K-Neighbors Classifier': KNeighborsClassifier()
}

best_estimators = {}

# Оптимизация гиперпараметров с помощью Grid Search
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_estimators[name] = grid_search.best_estimator_
    print(f"{name} best parameters: {grid_search.best_params_}")
    print(f"{name} best score: {grid_search.best_score_:.4f}")


### 3. Обучение и оценка моделей

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Обучение и оценка моделей
for name, clf in best_estimators.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Вывод матрицы истинности с пояснениями
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print("Confusion Matrix Explanation:")
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i == j:
                print(f"True Positive (TP) for class {i}: {conf_matrix[i, j]} - correctly predicted positives for class {i}")
            else:
                print(f"False Negative (FN) for class {i}, predicted as class {j}: {conf_matrix[i, j]} - incorrectly predicted as class {j}")    


from sklearn.ensemble import VotingClassifier

# Создание ансамбля моделей
ensemble_clf = VotingClassifier(estimators=[
    ('log_reg', best_estimators['Logistic Regression']),
    ('rf', best_estimators['Random Forest']),
    ('gb', best_estimators['Gradient Boosting']),
    ('svm', best_estimators['Support Vector Machine']),
    ('knn', best_estimators['K-Neighbors Classifier'])
], voting='hard')

# Обучение ансамбля моделей
ensemble_clf.fit(X_train, y_train)
ensemble_y_pred = ensemble_clf.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_y_pred)
print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
print(classification_report(y_test, ensemble_y_pred))

# Вывод матрицы истинности для ансамбля моделей с пояснениями
conf_matrix = confusion_matrix(y_test, ensemble_y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
plt.title('Confusion Matrix for Ensemble Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Confusion Matrix Explanation:")
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        if i == j:
            print(f"True Positive (TP) for class {i}: {conf_matrix[i, j]} - correctly predicted positives for class {i}")
        else:
            print(f"False Negative (FN) for class {i}, predicted as class {j}: {conf_matrix[i, j]} - incorrectly predicted as class {j}")                


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest_anomalies = iso_forest.fit_predict(X)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_anomalies = lof.fit_predict(X)

# OneClass SVM
one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
one_class_svm_anomalies = one_class_svm.fit_predict(X)

# Вывод результатов поиска аномалий
print(f"Isolation Forest Anomalies: {iso_forest_anomalies}")
print(f"Local Outlier Factor Anomalies: {lof_anomalies}")
print(f"OneClass SVM Anomalies: {one_class_svm_anomalies}")

# Визуализация результатов аномалий
def plot_anomalies(X, anomalies, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=anomalies, cmap='viridis', marker='.')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Визуализация аномалий для Isolation Forest
plot_anomalies(pd.DataFrame(X), iso_forest_anomalies, 'Isolation Forest Anomalies')

# Визуализация аномалий для Local Outlier Factor
plot_anomalies(pd.DataFrame(X), lof_anomalies, 'Local Outlier Factor Anomalies')

# Визуализация аномалий для OneClass SVM
plot_anomalies(pd.DataFrame(X), one_class_svm_anomalies, 'OneClass SVM Anomalies')            