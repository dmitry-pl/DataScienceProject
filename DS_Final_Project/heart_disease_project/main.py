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


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Инициализация моделей
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'K-Neighbors Classifier': KNeighborsClassifier()
}

# Обучение и оценка моделей
for name, clf in classifiers.items():
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



### 3. Создание ансамбля моделей

from sklearn.ensemble import VotingClassifier

# Создание ансамбля моделей
ensemble_clf = VotingClassifier(estimators=[
    ('log_reg', classifiers['Logistic Regression']),
    ('rf', classifiers['Random Forest']),
    ('gb', classifiers['Gradient Boosting']),
    ('svm', classifiers['Support Vector Machine']),
    ('knn', classifiers['K-Neighbors Classifier'])
], voting='hard')

# Обучение ансамбля моделей
ensemble_clf.fit(X_train, y_train)
ensemble_y_pred = ensemble_clf.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_y_pred)
print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")
print(classification_report(y_test, ensemble_y_pred))

# Вывод матрицы истинности для ансамбля моделей с пояснениями
conf_matrix = confusion_matrix(y_test, ensemble_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix for Ensemble Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("Confusion Matrix Explanation:")
print(f"True Negative (TN): {conf_matrix[0, 0]} - correctly predicted negatives (реально здоровые и предсказаны как здоровые)")
print(f"False Positive (FP): {conf_matrix[0, 1]} - incorrectly predicted as positive (реально здоровые, но предсказаны как больные)")
print(f"False Negative (FN): {conf_matrix[1, 0]} - incorrectly predicted as negative (реально больные, но предсказаны как здоровые)")
print(f"True Positive (TP): {conf_matrix[1, 1]} - correctly predicted positives (реально больные и предсказаны как больные)")    


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

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




# Визуализация ошибок прогнозирования

# Logistic Regression
log_reg_clf = classifiers['Logistic Regression']
log_reg_y_pred = log_reg_clf.predict(X_test)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=log_reg_y_pred)
plt.title('Logistic Regression Predictions')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()