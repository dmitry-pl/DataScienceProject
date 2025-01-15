import pandas as pd

class DataProcessing:
    def __init__(self, data):
        self.data = data

    def count_missing_values(self):
        return self.data.isnull().sum()

    def report_missing_values(self):
        missing_values = self.count_missing_values()
        report = pd.DataFrame({'Столбец': missing_values.index, 'Пропущенные значения': missing_values.values})
        return report

    def fill_missing_values(self, method='mean'):
        if method == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif method == 'median':
            self.data = self.data.fillna(self.data.median())
        elif method == 'mode':
            self.data = self.data.fillna(self.data.mode().iloc[0])

        self.data = self.data.infer_objects(copy=False)
        return self.data