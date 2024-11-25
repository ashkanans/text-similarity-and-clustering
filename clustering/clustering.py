from time import time

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


class Clustering:
    """Performs clustering and visualization."""

    @staticmethod
    def elbow_method(data, max_clusters=10):
        """Apply the elbow method to find the optimal number of clusters."""
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++")
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        plt.figure()
        plt.plot(range(1, max_clusters + 1), inertias, marker='o')
        plt.title("Elbow Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.grid()
        plt.show()

    @staticmethod
    def perform_clustering(data, algorithm="kmeans", n_clusters=3, **kwargs):
        """Perform clustering and return labels and silhouette score."""
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++")
        elif algorithm == "dbscan":
            model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
        else:
            raise ValueError("Unsupported clustering algorithm.")

        start_time = time()
        labels = model.fit_predict(data)
        end_time = time()
        silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else -1

        return labels, silhouette, end_time - start_time

    @staticmethod
    def visualize_clusters(data, labels, title="Cluster Visualization"):
        """Visualize clusters using the first two dimensions."""
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar()
        plt.show()
