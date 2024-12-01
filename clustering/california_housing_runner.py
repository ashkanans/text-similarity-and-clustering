import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


class CaliforniaHousingRunner:
    """Runs the entire pipeline for California Housing dataset."""

    def __init__(self, data_handler, feature_engineer, clustering, score_method="silhouette"):
        """
        Initialize the runner with dependencies and score method.
        """
        self.score_method = score_method
        self.data_handler = data_handler
        self.feature_engineer = feature_engineer
        self.clustering = clustering
        self.data = None
        self.engineered_data = None

    def run(self, actions):
        """
        Execute the pipeline actions specified in the input.

        :param actions: List of actions to perform, e.g., "download", "load", "clustering_raw".
        """
        for action in actions:
            if action == "download":
                self.download()
            elif action == "load":
                self.load()
            elif action == "clustering_raw":
                labels, numeric_data, optimal_clusters = self.perform_clustering(raw=True)
                self.compare_and_visualize(numeric_data=numeric_data, labels=labels)
            elif action == "clustering_engineered":
                self.feature_engineering()
                labels, numeric_data, optimal_clusters = self.perform_clustering(raw=False)
                self.compare_and_visualize(numeric_data=numeric_data, labels=labels)
            elif action == "default":
                # if the default actions was given, load and perform a complete comparison pipeline
                self.load()
                labels_raw, data_raw, optimal_clusters_raw = self.perform_clustering(raw=True)

                # Perform PCA and visualize clusters for raw data
                print("Visualizing Raw Data Clusters:")
                self.visualize_pca(data_raw, labels_raw, title="Raw Data", n_components=2)
                self.visualize_pca(data_raw, labels_raw, title="Raw Data", n_components=3)

                # Feature engineer the data and perform clustering
                self.feature_engineering()
                labels_eng, data_eng, optimal_clusters_eng = self.perform_clustering(raw=False)

                # Perform PCA and visualize clusters for feature-engineered data
                print("Visualizing Feature-Engineered Data Clusters:")
                self.visualize_pca(data_eng, labels_eng, title="Feature-Engineered Data", n_components=2)
                self.visualize_pca(data_eng, labels_eng, title="Feature-Engineered Data", n_components=3)

                # Compare results
                print("\nClustering Comparison:")
                print(f"Raw Data - Optimal Clusters: {optimal_clusters_raw}")
                print(f"Engineered Data - Optimal Clusters: {optimal_clusters_eng}")
            else:
                raise ValueError(f"Unknown action: {action}")


    def download(self):
        """
        Download the dataset using the data handler.
        """
        self.data_handler.download_data()

    def load(self):
        """
        Load and clean the dataset.
        """
        print("Loading dataset...")
        self.data = self.data_handler.load_data()
        print(f"Dataset shape before cleaning: {self.data.shape}")
        self.data = self.data_handler.clean_data(self.data)
        print(f"Dataset shape after cleaning: {self.data.shape}")

    def perform_clustering(self, raw=True):
        """
        Perform clustering on either raw or feature-engineered data.

        Parameters:
            raw (bool): If True, clusters raw data; otherwise, clusters feature-engineered data.

        Returns:
            tuple: Cluster labels, data used for clustering, and the number of optimal clusters.
        """
        # Select appropriate data for clustering
        if raw:
            self._validate_data(self.data, "Raw data must be loaded first.")
            # Retain only numeric columns and drop rows with missing values for clustering
            clustering_data = self.data.select_dtypes(include=[np.number]).dropna()
        else:
            self._validate_data(self.engineered_data, "Engineered data must be loaded first.")
            clustering_data = self.engineered_data

        # Determine optimal clusters
        optimal_clusters = self.clustering.find_optimal_clusters(
            clustering_data, method=self.score_method
        )

        # Perform clustering
        start_time = time.time()
        labels, score, time_taken = self.clustering.perform_clustering(
            clustering_data, algorithm="kmeans", n_clusters=optimal_clusters, score_method=self.score_method
        )
        end_time = time.time()

        # Log clustering duration
        print(f"Clustering took {end_time - start_time:.2f} seconds on {'raw' if raw else 'engineered'} data")

        # Display results
        self._print_clustering_results(raw, optimal_clusters, score, time_taken)

        return labels, clustering_data, optimal_clusters

    def _validate_data(self, data, error_message):
        """
        Validate if the data is loaded.

        Parameters:
            data: Data to be validated.
            error_message (str): Error message if validation fails.

        Raises:
            ValueError: If data is None.
        """
        if data is None:
            raise ValueError(error_message)

    def _print_clustering_results(self, raw, optimal_clusters, score, time_taken):
        """
        Print clustering results in a consistent format.

        Parameters:
            raw (bool): Indicates if clustering was performed on raw data.
            optimal_clusters (int): The optimal number of clusters found.
            score (float): The clustering score.
            time_taken (float): The time taken for clustering in seconds.
        """
        data_format = "Raw" if raw else "Engineered"
        print(f"{data_format} Data Clustering Results:")
        print(f"- Optimal Clusters: {optimal_clusters}")
        print(f"- {self.score_method.title()} Score: {score:.4f}")
        print(f"- Time Taken: {time_taken:.2f} seconds")

    def feature_engineering(self):
        """
        Perform feature engineering on the dataset.
        """
        if self.data is None:
            raise ValueError("Data must be loaded first.")
        self.engineered_data = self.feature_engineer.preprocess_data(self.data,
                                                                     scale_numeric=True,
                                                                     one_hot_encode=True,
                                                                     add_interaction_features=True,
                                                                     log_transform_features=False,
                                                                     discretize_age=True,
                                                                     handle_missing=True)

    def compare_and_visualize(self, numeric_data, labels):
        """
        Compare clustering metrics and visualize results based on the selected score method.
        """
        print("\n--- Comparing and Visualizing Results ---")

        # Compute and display the selected metric
        self.clustering.compute_metrics(numeric_data, labels, score_method=self.score_method)

        # Tailored visualizations based on the score method
        if self.score_method == "silhouette":
            print(f"Silhouette Score is being used for analysis.")
            self.clustering.silhouette_distribution(numeric_data, labels)
            self.clustering.cluster_heatmap(numeric_data.copy(), labels)
            self.clustering.pairplot_clusters(numeric_data.copy(), labels)

        elif self.score_method == "davies_bouldin":
            print(f"Davies-Bouldin Index is being used for analysis.")
            self.clustering.cluster_heatmap(numeric_data.copy(), labels)
            self.clustering.pairplot_clusters(numeric_data.copy(), labels)

        elif self.score_method == "calinski_harabasz":
            print(f"Calinski-Harabasz Index is being used for analysis.")
            self.clustering.cluster_heatmap(numeric_data.copy(), labels)
            self.clustering.pairplot_clusters(numeric_data.copy(), labels)

        elif self.score_method == "elbow":
            print(f"Inertia (Elbow Method) is being used for analysis.")
            # Visualizations relevant to Elbow
            print("The Elbow Graph is typically used to determine the optimal number of clusters.")
            self.clustering.cluster_heatmap(numeric_data.copy(), labels)
            self.clustering.pairplot_clusters(numeric_data.copy(), labels)

        else:
            raise ValueError("Unsupported score method for visualization.")

        print("\n--- Comparison and Visualization Complete ---")

    @staticmethod
    def visualize_pca(data, labels, title="PCA Visualization", n_components=2):
        """
        Perform PCA and visualize the data in 2D or 3D.

        Parameters:
            data (DataFrame or ndarray): The input data for PCA.
            labels (array-like): Cluster labels for coloring the points.
            title (str): Title for the plot.
            n_components (int): Number of dimensions to reduce the data to (2 or 3).
        """
        pca = PCA(n_components=n_components)
        data_reduced = pca.fit_transform(data)

        if n_components == 2:
            plt.figure()
            plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis', s=50)
            plt.title(f"{title} (2D)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.colorbar(label="Cluster")
            plt.show(block=True)
        elif n_components == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                data_reduced[:, 0],
                data_reduced[:, 1],
                data_reduced[:, 2],
                c=labels,
                cmap='viridis',
                s=50
            )
            ax.set_title(f"{title} (3D)")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_zlabel("Principal Component 3")
            fig.colorbar(scatter, label="Cluster")
            plt.show(block=True)
        else:
            raise ValueError("n_components must be 2 or 3.")
