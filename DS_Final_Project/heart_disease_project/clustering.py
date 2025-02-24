import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=5, random_state=42)

    def segment_data(self, data):
        kmeans_clusters = self.kmeans.fit_predict(data.drop('num', axis=1))
        return kmeans_clusters

    def plot_clusters(self, data, clusters):
        X_df = pd.DataFrame(data.drop('num', axis=1), columns=data.columns[:-1])
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_df.iloc[:, 0], X_df.iloc[:, 1], c=clusters, cmap='viridis', marker='.')
        plt.title('Кластеры KMeans по меткам target')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        legend1 = plt.legend(*scatter.legend_elements(), title="Кластеры")
        plt.gca().add_artist(legend1)
        plt.show()