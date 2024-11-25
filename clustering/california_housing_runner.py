import numpy as np


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
                labels, numeric_data = self.clustering_raw()
                self.compare_and_visualize(numeric_data=numeric_data, labels=labels)
            elif action == "clustering_engineered":
                self.feature_engineering()
                self.clustering_engineered()
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

    def clustering_raw(self):
        """
        Perform clustering on raw data using the selected scoring method.
        """
        if self.data is None:
            raise ValueError("Data must be loaded first.")

        # Prepare numeric data
        numeric_data = self.data.select_dtypes(include=[np.number]).dropna()

        # Determine optimal clusters
        optimal_clusters = self.clustering.find_optimal_clusters(
            numeric_data, method=self.score_method
        )

        labels, score, time_taken = self.clustering.perform_clustering(
            numeric_data, algorithm="kmeans", n_clusters=optimal_clusters, score_method=self.score_method
        )

        print("Clustering Results for Raw Data:")
        print(f"- Optimal Clusters: {optimal_clusters}")
        if self.score_method == "elbow":
            print(f"- Inertia (Elbow Method): {score:.2f}")  # Explicitly mention inertia
        else:
            print(f"- {self.score_method.title()} Score: {score:.4f}")
        print(f"- Time Taken: {time_taken:.2f} seconds")
        return labels, numeric_data

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
                                                                     discretize_age=False,
                                                                     handle_missing=True)

    def clustering_engineered(self):
        """
        Perform clustering on feature-engineered data.
        """
        if self.engineered_data is None:
            raise ValueError("Feature-engineered data must be prepared first.")

        # Determine optimal clusters for engineered data
        optimal_clusters = self.clustering.find_optimal_clusters(
            self.engineered_data, method=self.score_method
        )

        # Perform clustering
        labels, score, time_taken = self.clustering.perform_clustering(
            self.engineered_data, algorithm="kmeans", n_clusters=optimal_clusters, score_method=self.score_method
        )

        # Print results
        print("Engineered Data Clustering Results:")
        print(f"- Optimal Clusters: {optimal_clusters}")
        print(f"- {self.score_method.title()} Score: {score:.4f}")
        print(f"- Time Taken: {time_taken:.2f} seconds")

        # Visualization
        self.compare_and_visualize(self.engineered_data, labels)

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
