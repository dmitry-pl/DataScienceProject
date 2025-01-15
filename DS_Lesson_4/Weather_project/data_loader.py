import requests
import pandas as pd
from datetime import datetime

class DataLoader:
    def __init__(self, api_key, city):
        self.api_key = api_key
        self.city = city
        self.url = f"http://api.openweathermap.org/data/2.5/forecast"

    def load_data(self):
        data = pd.DataFrame(columns=['Дата', 'Температура', 'Влажность', 'Ветер', 'Направление'])

        params = {
            'q': self.city,
            'appid': self.api_key,
            'units': 'metric'
        }

        response = requests.get(self.url, params=params)
        if response.status_code == 200:
            forecast = response.json().get('list', [])
            for item in forecast:
                date = datetime.utcfromtimestamp(item['dt'])
                temp = item['main']['temp']
                humidity = item['main']['humidity']
                wind_speed = item['wind']['speed']
                wind_direction = item['wind']['deg']

                new_row = pd.DataFrame({
                    'Дата': [date],
                    'Температура': [temp],
                    'Влажность': [humidity],
                    'Ветер': [wind_speed],
                    'Направление': [wind_direction]
                })

                # Исключаем пустые или все-NaN записи
                if not new_row.dropna(how='all').empty:
                    data = pd.concat([data, new_row], ignore_index=True, sort=False)

        else:
            print("Ошибка при запросе данных:", response.status_code)

        return data