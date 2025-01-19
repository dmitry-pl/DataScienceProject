import pandas as pd

class DataProcessing:
    def __init__(self, data):  # Инициализация с передачей DataFrame
        self.data = data

    def count_missing_values(self):
        # Подсчёт пропущенных значений в каждом столбце
        return self.data.isnull().sum()

    def report_missing_values(self):
        # Создание отчёта о пропущенных значениях
        missing_values = self.count_missing_values()
        report = pd.DataFrame({'Столбец': missing_values.index, 'Пропущенные значения': missing_values.values})
        return report

    def fill_missing_values(self, data, method='mean'):
        # Заполнение пропущенных значений заданным методом (среднее, медиана, мода)
        if method == 'mean':
            data = data.fillna(data.mean())
        elif method == 'median':
            data = data.fillna(data.median())
        elif method == 'mode':
            data = data.fillna(data.mode().iloc[0])

        data = data.infer_objects(copy=False)  # Преобразование типов данных после заполнения пропущенных значений
        return data