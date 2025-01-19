import pymysql
import sys

class DatabaseManager:
    def __init__(self, host, user, password):
        self.host = host
        self.user = user
        self.password = password
        self.db = 'TitanicDB'
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = pymysql.connect(host=self.host, port=3306, user=self.user, password=self.password)
            self.cursor = self.connection.cursor()
            print("Успешное подключение к серверу MySQL")

        except pymysql.MySQLError as e:
            print(f"Ошибка при подключении к серверу MySQL: {e}")
            sys.exit()

    def create_database(self):
        try:
            if self.cursor is None:
                raise AttributeError("Соединение с сервером MySQL не установлено. Проверьте параметры подключения.")
            self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db};")
            print("База данных создана или уже существует")
        except pymysql.MySQLError as e:
            print(f"Ошибка при создании базы данных: {e}")

    def connect_to_db(self):
        try:
            self.connection = pymysql.connect(host=self.host, port=3306, user=self.user, password=self.password, db=self.db)
            self.cursor = self.connection.cursor()
            print(f"Успешное подключение к базе данных {self.db}")

        except pymysql.MySQLError as e:
            print(f"Ошибка при подключении к базе данных {self.db}: {e}")
            sys.exit()

    def create_tables(self):
        create_passengers_table = """
        CREATE TABLE IF NOT EXISTS Passengers (
            PassengerId INT PRIMARY KEY,
            Name VARCHAR(255),
            Sex VARCHAR(10),
            Age FLOAT,
            SibSp INT,
            Parch INT,
            Ticket VARCHAR(20),
            Fare FLOAT,
            Cabin VARCHAR(20),
            Embarked VARCHAR(1),
            Survived INT
        );
        """
        try:
            if self.cursor is None:
                raise AttributeError("Соединение с базой данных не установлено. Проверьте параметры подключения.")
            self.cursor.execute(create_passengers_table)
            self.connection.commit()
            print("Таблицы созданы успешно")
        except pymysql.MySQLError as e:
            print(f"Ошибка при создании таблиц: {e}")

    def insert_data(self, data):
        try:
            if self.cursor is None:
                raise AttributeError("Соединение с базой данных не установлено. Проверьте параметры подключения.")
            for i, row in data.iterrows():
                sql = """
                INSERT INTO Passengers (PassengerId, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, Survived)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                self.cursor.execute(sql, tuple(row))
            self.connection.commit()
            print("Данные успешно загружены в таблицы")
        except pymysql.MySQLError as e:
            print(f"Ошибка при вставке данных: {e}")

    def close(self):
        if self.connection:
            self.connection.close()
            print("Соединение с базой данных закрыто")