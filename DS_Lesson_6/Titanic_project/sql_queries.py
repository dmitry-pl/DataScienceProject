class SQLQueries:
    def __init__(self, cursor):
        self.cursor = cursor

    def execute_query(self, query):
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            print(f"Ошибка при выполнении запроса: {e}")
            return None

    def get_max_fare(self):
        query = "SELECT MAX(Fare) FROM Passengers;"
        return self.execute_query(query)

    def get_min_age(self):
        query = "SELECT MIN(Age) FROM Passengers;"
        return self.execute_query(query)

    def get_avg_age(self):
        query = "SELECT AVG(Age) FROM Passengers;"
        return self.execute_query(query)

    def get_survival_counts(self):
        query = "SELECT Survived, COUNT(*) FROM Passengers GROUP BY Survived;"
        return self.execute_query(query)