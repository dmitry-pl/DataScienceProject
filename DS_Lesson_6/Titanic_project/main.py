from titanic_dataset import TitanicDataset
from database_manager import DatabaseManager
from sql_queries import SQLQueries
from data_visualizer import DataVisualizer

# Шаг 1: Загрузка данных
url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
titanic_dataset = TitanicDataset(url)
titanic_dataset.load_data()
data = titanic_dataset.get_data()

# Шаг 2: Работа с базой данных
db_manager = DatabaseManager(host='localhost', user='root', password='1111')
db_manager.connect()
db_manager.create_database()
db_manager.connect_to_db()  # Подключаемся к созданной базе данных
db_manager.create_tables()
db_manager.insert_data(data)

# Шаг 3: Выполнение SQL запросов
sql_queries = SQLQueries(db_manager.cursor)
print("Максимальная стоимость билета:", sql_queries.get_max_fare())
print("Минимальный возраст пассажира:", sql_queries.get_min_age())
print("Средний возраст пассажиров:", sql_queries.get_avg_age())
print("Количество выживших и погибших пассажиров:", sql_queries.get_survival_counts())

# Шаг 4: Визуализация данных
visualizer = DataVisualizer(data)
visualizer.plot_survival_counts()
visualizer.plot_age_distribution()

# Закрытие соединения с базой данных
db_manager.close()