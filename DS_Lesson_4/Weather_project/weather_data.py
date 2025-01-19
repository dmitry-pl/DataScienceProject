import requests
import pandas as pd
from datetime import datetime, timedelta

def get_weather_data(api_key, location, start_date, end_date):
    # Функция для получения данных о погоде за указанный день
    def fetch_weather_data(date):
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date}"
        response = requests.get(url)
        return response.json()

    # Данные за указанный период
    weather_data = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    while current_date.strftime('%Y-%m-%d') <= end_date:
        data = fetch_weather_data(current_date.strftime('%Y-%m-%d'))
        if 'forecast' in data and 'forecastday' in data['forecast']:
            for hour in data['forecast']['forecastday'][0]['hour']:
                weather_data.append({
                    'Дата': hour['time'],
                    'Температура': hour['temp_c'],
                    'Ощущается': hour['feelslike_c'],
                    'Влажность': hour['humidity'],
                    'Ветер': hour['wind_kph'],
                    'Направление': hour['wind_degree'],
                    'Осадки': hour['precip_mm'],
                    'Облачность': hour['cloud']
                    })
        current_date += timedelta(days=1)
    
    # Преобразование данных в DataFrame
    df = pd.DataFrame(weather_data)
    return df