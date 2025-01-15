#Этот модуль отвечает за создание и удаление различных типов визуализаций.
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self):
        self.figures = {}

    def add_histogram(self, data, column, title):
        fig, ax = plt.subplots()
        ax.hist(data[column])
        ax.set_title(title)
        self.figures[title] = fig

    def add_line_chart(self, data, x_column, y_column, title):
        fig, ax = plt.subplots()
        ax.plot(data[x_column], data[y_column])
        ax.set_title(title)
        self.figures[title] = fig

    def add_scatter_plot(self, data, x_column, y_column, title):
        fig, ax = plt.subplots()
        ax.scatter(data[x_column], data[y_column])
        ax.set_title(title)
        self.figures[title] = fig

    def remove_plot(self, title):
        if title in self.figures:
         del self.figures[title]

### Объяснение
#- **Импорт библиотеки**: Используется `matplotlib.pyplot` для создания визуализаций.
#- **Класс `Visualization`**: Инициализирует пустой словарь для хранения графиков.
#- **Методы**:
#  - `add_histogram`: Создает и сохраняет гистограмму.
#  - `add_line_chart`: Создает и сохраняет линейный график.
#  - `add_scatter_plot`: Создает и сохраняет диаграмму рассеяния.
#  - `remove_plot`: Удаляет график по названию.
        