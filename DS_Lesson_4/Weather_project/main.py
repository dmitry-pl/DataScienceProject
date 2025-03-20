# main.py
from data_loader import DataLoader
from visualization import Visualization
from data_processing import DataProcessing
from prediction import Prediction
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt  # Добавляем импорт библиотеки matplotlib.pyplot

API_KEY = "3af83f3cd691492d472220ce62e52f52"
CITY = "Москва"
TARGET_COLUMN = "Температура"

# Загрузка данных
loader = DataLoader(API_KEY, CITY)
data = loader.load_data()
#print(data)

# Проверка наличия данных
if data.empty:
    print("Данные не загружены.")
else:
    # Обработка данных
    processor = DataProcessing(data)
    missing_report = processor.report_missing_values()
    print(missing_report)
    data = processor.fill_missing_values(method='mean')

    # Преобразование столбца с датой в отдельные компоненты
    data['Год'] = data['Дата'].dt.year
    data['Месяц'] = data['Дата'].dt.month
    data['День'] = data['Дата'].dt.day
    data.drop(columns=['Дата'], inplace=True)
    #print(data)

    # Проверка данных после преобразования даты
    print("Данные после преобразования даты:\n", data.head())

    # Визуализация данных
    viz = Visualization()
    viz.add_histogram(data, 'Температура', 'Гистограмма температуры')
    viz.add_line_chart(data, 'День', 'Температура', 'Линейный график температуры')
    viz.add_scatter_plot(data, 'Температура', 'Влажность', 'Диаграмма рассеяния: Температура vs Влажность')

    # Прогнозирование температур
    predictor = Prediction(data, TARGET_COLUMN)
    X, y = predictor.preprocess_data()
    print("Признаки (X):\n", X.head())
    print("Целевая переменная (y):\n", y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = predictor.train_model(X_train, y_train)
    y_pred = predictor.predict(model, X_test)

    # Визуализация истинных и предсказанных значений
    predictor.plot_predictions(y_test, y_pred)

    # Оценка модели
    mse, r2 = predictor.evaluate_model(y_test, y_pred)
    print(f"Среднеквадратичная ошибка: {mse:.2f}")
    print(f"Коэффициент детерминации R^2: {r2:.2f}")

    # Показ всех графиков
    #for fig in viz.figures.values():
    #    plt.show()