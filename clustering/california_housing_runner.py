import numpy as np


class CaliforniaHousingRunner:
    """Runs the entire pipeline."""

    def __init__(self, data_handler, feature_engineer, clustering):
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        self.clustering = clustering
        self.data = None
        self.engineered_data = None

    def run(self, actions):
        """Execute specified actions."""
        for action in actions:
            if action == "download":
                self.data_handler.download_data()
            elif action == "load":
                self.data = self.data_handler.load_data()
                print(f"Dataset shape: {self.data.shape}")
            elif action == "elbow_raw":
                if self.data is None:
                    raise ValueError("Data must be loaded first.")
                numeric_data = self.data.select_dtypes(include=[np.number]).dropna()
                self.clustering.elbow_method(numeric_data)
            elif action == "clustering_raw":
                if self.data is None:
                    raise ValueError("Data must be loaded first.")
                numeric_data = self.data.select_dtypes(include=[np.number]).dropna()
                labels, silhouette, time_taken = self.clustering.perform_clustering(numeric_data, algorithm="kmeans",
                                                                                    n_clusters=3)
                print(f"Raw Data Clustering: Silhouette Score = {silhouette}, Time = {time_taken:.2f} seconds")
            elif action == "feature_engineering":
                if self.data is None:
                    raise ValueError("Data must be loaded first.")
                self.engineered_data = self.feature_engineer.preprocess_data(self.data)
            elif action == "elbow_engineered":
                if self.engineered_data is None:
                    raise ValueError("Feature-engineered data must be prepared first.")
                self.clustering.elbow_method(self.engineered_data)
            elif action == "clustering_engineered":
                if self.engineered_data is None:
                    raise ValueError("Feature-engineered data must be prepared first.")
                labels, silhouette, time_taken = self.clustering.perform_clustering(self.engineered_data,
                                                                                    algorithm="kmeans", n_clusters=3)
                print(f"Engineered Data Clustering: Silhouette Score = {silhouette}, Time = {time_taken:.2f} seconds")
