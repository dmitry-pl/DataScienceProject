import pandas as pd

class TitanicDataset:
    def __init__(self, url):
        self.url = url
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.url)
            print("Данные успешно загружены")
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")

    def get_data(self):
        return self.data