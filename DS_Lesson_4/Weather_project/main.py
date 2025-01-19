from weather_data import get_weather_data
from datetime import datetime, timedelta
from visualization import Visualization
from data_processing import DataProcessing
from prediction import Prediction
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Устанавливаем API-ключ и параметры
api_key = "dbef3b017e6042a09f2162813251601"
location = "Минск"
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

# Получение данных о погоде
data = get_weather_data(api_key, location, start_date, end_date)

# Проверка наличия данных
if data.empty:
    print("Данные не загружены.")
else:
    # Преобразование столбца с датой в datetime формат перед обработкой данных
    data['Дата'] = pd.to_datetime(data['Дата'])
    
    # Обработка данных
    processor = DataProcessing(data)
    missing_report = processor.report_missing_values()
    print(missing_report)
    # Исключаем столбец 'Дата' при заполнении пропущенных значений
    data_excl_date = data.drop(columns=['Дата'])
    data_filled = processor.fill_missing_values(data_excl_date, method='mean')
    data[data_excl_date.columns] = data_filled[data_excl_date.columns]  # Обновляем исходные данные
    
    # Преобразование столбца с датой в отдельные компоненты
    data['Год'] = data['Дата'].dt.year
    data['Месяц'] = data['Дата'].dt.month
    data['День'] = data['Дата'].dt.day
    data.drop(columns=['Дата'], inplace=True)

    # Проверка данных после преобразования даты
    print("Данные после преобразования даты:\n", data.head())

    # Визуализация данных
    viz = Visualization()
    viz.add_histogram(data, 'Температура', 'Гистограмма температуры')
    viz.add_line_chart(data, 'День', 'Температура', 'Линейный график температуры')
    viz.add_scatter_plot(data, 'Температура', 'Влажность', 'Диаграмма рассеяния: Температура vs Влажность')

    # Прогнозирование температур
    TARGET_COLUMN = "Температура"
    predictor = Prediction(data, TARGET_COLUMN)
    X, y = predictor.preprocess_data()
    print("Признаки (X):\n", X.head())
    print("Целевая переменная (y):\n", y.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = predictor.train_model(X_train, y_train)
    y_pred = predictor.predict(model, X_test)

    # Визуализация истинных и предсказанных значений на примере 50-ти элементов
    predictor.plot_predictions(y_test, y_pred)

    # Показ всех графиков
    viz.show_all()