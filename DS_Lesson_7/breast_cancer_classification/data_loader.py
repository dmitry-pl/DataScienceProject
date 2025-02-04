import pandas as pd
from sklearn.datasets import load_breast_cancer

class DataLoader:
    def __init__(self):
        self.data = None

    def load_data(self):
        """
        Загружает и обрабатывает данные из набора данных Breast Cancer Wisconsin.
        """
        # Загружаем набор данных Breast Cancer Wisconsin
        cancer_data = load_breast_cancer()
        self.data = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
        self.data['target'] = cancer_data.target
        print("Заголовки на английском:")
        print(self.data.columns.tolist())

        # Переводим заголовки на русский
        columns_translation = {
            'mean radius': 'Средний радиус', 
            'mean texture': 'Средняя текстура', 
            'mean perimeter': 'Средний периметр', 
            'mean area': 'Средняя площадь',
            'mean smoothness': 'Средняя гладкость', 
            'mean compactness': 'Средняя компактность', 
            'mean concavity': 'Средняя вогнутость', 
            'mean concave points': 'Средние вогнутые точки', 
            'mean symmetry': 'Средняя симметрия', 
            'mean fractal dimension': 'Средняя фрактальная размерность',
            'radius error': 'Ошибка радиуса', 
            'texture error': 'Ошибка текстуры', 
            'perimeter error': 'Ошибка периметра', 
            'area error': 'Ошибка площади', 
            'smoothness error': 'Ошибка гладкости', 
            'compactness error': 'Ошибка компактности', 
            'concavity error': 'Ошибка вогнутости', 
            'concave points error': 'Ошибка вогнутых точек', 
            'symmetry error': 'Ошибка симметрии', 
            'fractal dimension error': 'Ошибка фрактальной размерности',
            'worst radius': 'Наихудший радиус', 
            'worst texture': 'Наихудшая текстура', 
            'worst perimeter': 'Наихудший периметр', 
            'worst area': 'Наихудшая площадь', 
            'worst smoothness': 'Наихудшая гладкость', 
            'worst compactness': 'Наихудшая компактность', 
            'worst concavity': 'Наихудшая вогнутость', 
            'worst concave points': 'Наихудшие вогнутые точки', 
            'worst symmetry': 'Наихудшая симметрия', 
            'worst fractal dimension': 'Наихудшая фрактальная размерность',
            'target': 'Целевая переменная'
        }
        self.data.columns = [columns_translation.get(col, col) for col in self.data.columns]

        print("Заголовки на русском:")
        print(self.data.columns.tolist())

        print("\nТип данных каждого столбца:")
        print(self.data.dtypes)

        print("\nПервые 5 строк:")
        print(self.data.head())

        return self.data

    def get_features_and_target(self):
        """
        Возвращает признаки (X) и целевую переменную (y) из загруженного набора данных.
        """
        if self.data is None:
            self.load_data()
        X = self.data.drop(columns=['Целевая переменная'])
        y = self.data['Целевая переменная']
        return X, y