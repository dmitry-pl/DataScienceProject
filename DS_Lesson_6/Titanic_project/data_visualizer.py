import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_survival_counts(self):
        survived_counts = self.data['Survived'].value_counts()
        plt.figure(figsize=(8, 6))
        survived_counts.plot(kind='bar')
        plt.title('Соотношение выживших и погибших пассажиров')
        plt.xlabel('Выжил')
        plt.ylabel('Количество')
        plt.show()

    def plot_age_distribution(self):
        plt.figure(figsize=(8, 6))
        for survived in [0, 1]:
            plt.hist(self.data[self.data['Survived'] == survived]['Age'].dropna(), alpha=0.5, label=f'Выжил {survived}')
        plt.title('Возраст выживших и погибших пассажиров')
        plt.xlabel('Возраст')
        plt.ylabel('Количество')
        plt.legend()
        plt.show()