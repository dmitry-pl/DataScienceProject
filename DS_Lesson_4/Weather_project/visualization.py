import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):  # Инициализация с пустым словарём для хранения графиков
        self.figures = {}

    def add_histogram(self, data, column, title):
        # Создание гистограммы и добавление её в словарь
        fig, ax = plt.subplots()
        ax.hist(data[column])
        ax.set_title(title)
        self.figures[title] = fig

    def add_line_chart(self, data, x_column, y_column, title):
        # Создание линейного графика и добавление его в словарь
        fig, ax = plt.subplots()
        ax.plot(data[x_column], data[y_column])
        ax.set_title(title)
        self.figures[title] = fig

    def add_scatter_plot(self, data, x_column, y_column, title):
        # Создание диаграммы рассеяния и добавление её в словарь
        fig, ax = plt.subplots()
        ax.scatter(data[x_column], data[y_column])
        ax.set_title(title)
        self.figures[title] = fig

    def remove_plot(self, title):
        # Удаление графика по названию из словаря
        if title in self.figures:
            del self.figures[title]

    def show_all(self):  # Новая функция для отображения всех графиков
        for fig in self.figures.values():
            plt.figure(fig.number)
            plt.show()