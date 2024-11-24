from time import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples


class Clustering:
    """Performs clustering and visualization."""

    @staticmethod
    def find_optimal_clusters(data, method="silhouette", max_clusters=50):
        """
        Find the optimal number of clusters using the specified method.
        Available methods: 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'elbow'.
        """
        scores = []
        range_values = range(2, max_clusters + 1)  # Start from 2 clusters for most methods

        for k in range_values:
            kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++")
            labels = kmeans.fit_predict(data)

            if method == "silhouette":
                score = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
            elif method == "davies_bouldin":
                score = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else np.inf
            elif method == "calinski_harabasz":
                score = calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else 0
            elif method == "elbow":
                score = kmeans.inertia_
            else:
                raise ValueError(
                    "Unsupported method. Use 'silhouette', 'davies_bouldin', 'calinski_harabasz', or 'elbow'.")

            scores.append(score)

        # Determine optimal clusters based on the method
        if method == "elbow":
            # Use the second derivative to find the elbow point
            deltas = np.diff(scores)
            second_deltas = np.diff(deltas)
            optimal_clusters = range_values[np.argmin(second_deltas) + 1]  # Add 1 to adjust for indexing
        elif method in ["silhouette", "calinski_harabasz"]:
            # Higher scores are better for silhouette and calinski_harabasz
            optimal_clusters = range_values[scores.index(max(scores))]
        elif method == "davies_bouldin":
            # Lower scores are better for davies_bouldin
            optimal_clusters = range_values[scores.index(min(scores))]
        else:
            raise ValueError("Unsupported method. Use a valid method.")

        # Plot scores for visual analysis
        plt.figure()
        plt.plot(range_values, scores, marker='o')
        plt.title(f"Optimal Clusters using {method.replace('_', ' ').title()} Method")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Score")
        plt.grid()
        plt.show(block=True)

        print(f"Optimal number of clusters based on {method.replace('_', ' ').title()}: {optimal_clusters}")
        return optimal_clusters

    @staticmethod
    def silhouette_analysis(data, max_clusters=10):
        """Determine optimal clusters using Silhouette Score."""
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++")
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)

        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

        # Visualization
        plt.figure()
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.title("Silhouette Analysis")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.grid()
        plt.show(block=True)

        print(f"Optimal number of clusters based on Silhouette Score: {optimal_clusters}")
        return optimal_clusters

    @staticmethod
    def elbow_method(data, max_clusters=10):
        """Apply the Elbow Method to find the optimal number of clusters."""
        inertias = []
        range_values = range(1, max_clusters + 1)

        for k in range_values:
            kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++")
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)

        # Visualization
        plt.figure()
        plt.plot(range_values, inertias, marker='o')
        plt.title("Elbow Method for Optimal Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.grid()
        plt.show(block=True)

        print("Elbow Method graph displayed. Select optimal clusters visually.")
        return int(input("Enter the number of clusters based on the Elbow graph: "))

    @staticmethod
    def perform_clustering(data, algorithm="kmeans", n_clusters=3, score_method="silhouette", **kwargs):
        """
        Perform clustering and return labels, the chosen score (or inertia for 'elbow'), and execution time.
        """
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++")
        elif algorithm == "dbscan":
            model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
        else:
            raise ValueError("Unsupported clustering algorithm.")

        start_time = time()
        labels = model.fit_predict(data)
        end_time = time()

        # Compute the selected score or inertia for elbow
        if len(set(labels)) <= 1:  # Handle cases where clustering fails or only one cluster is found
            score = -1  # Invalid clustering
        elif score_method == "silhouette":
            score = silhouette_score(data, labels)
        elif score_method == "davies_bouldin":
            score = davies_bouldin_score(data, labels)
        elif score_method == "calinski_harabasz":
            score = calinski_harabasz_score(data, labels)
        elif score_method == "elbow":
            if algorithm != "kmeans":
                raise ValueError("Elbow method is only supported for KMeans.")
            score = model.inertia_  # Inertia is the metric used for the elbow method
        else:
            raise ValueError(
                "Unsupported score method. Choose 'silhouette', 'davies_bouldin', 'calinski_harabasz', or 'elbow'.")

        return labels, score, end_time - start_time

    @staticmethod
    def visualize_clusters(data, labels, title="Cluster Visualization"):
        """Visualize clusters using the first two dimensions."""
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar()
        plt.show(block=True)

    @staticmethod
    def compute_metrics(data, labels, score_method="silhouette"):
        """
        Compute and display clustering metrics based on the selected score method.

        Parameters:
        - data: Numeric data used for clustering.
        - labels: Cluster labels.
        - score_method: The metric to compute ('silhouette', 'davies_bouldin', 'calinski_harabasz', or 'elbow').
        """
        if len(set(labels)) <= 1:
            print("Invalid clustering: only one cluster found.")
            return

        if score_method == "silhouette":
            score = silhouette_score(data, labels)
            print(f"Silhouette Score: {score:.4f}")
        elif score_method == "davies_bouldin":
            score = davies_bouldin_score(data, labels)
            print(f"Davies-Bouldin Index: {score:.4f}")
        elif score_method == "calinski_harabasz":
            score = calinski_harabasz_score(data, labels)
            print(f"Calinski-Harabasz Index: {score:.4f}")
        elif score_method == "elbow":
            print("Elbow method does not compute a specific score after clustering.")
        else:
            raise ValueError(
                "Unsupported score method. Choose 'silhouette', 'davies_bouldin', 'calinski_harabasz', or 'elbow'.")

    @staticmethod
    def silhouette_distribution(data, labels):
        """Visualize silhouette score distribution."""
        silhouette_vals = silhouette_samples(data, labels)
        y_ticks = []
        y_lower, y_upper = 0, 0

        for cluster in np.unique(labels):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0)
            y_ticks.append((y_lower + y_upper) / 2)
            y_lower += len(cluster_silhouette_vals)

        plt.yticks(y_ticks, np.unique(labels))
        plt.ylabel("Cluster")
        plt.xlabel("Silhouette Score")
        plt.title("Silhouette Score Distribution")
        plt.show(block=True)

    @staticmethod
    def cluster_heatmap(data, labels):
        """Create a heatmap of feature means by cluster."""
        data['Cluster'] = labels
        cluster_means = data.groupby('Cluster').mean()

        sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Cluster Feature Heatmap")
        plt.show(block=True)

    @staticmethod
    def pairplot_clusters(data, labels, max_samples=10000, selected_features=None):
        """
        Efficient pairplot for clusters using Plotly.
        - Samples the data for large datasets.
        - Optionally focuses on selected features.
        """
        # Add cluster labels to the dataset
        data['Cluster'] = labels

        # Sample data if too large
        if len(data) > max_samples:
            data = data.sample(n=max_samples, random_state=42)

        # Focus on selected features if provided
        if selected_features:
            data = data[selected_features + ['Cluster']]

        # Convert Cluster column to string for categorical coloring in Plotly
        data['Cluster'] = data['Cluster'].astype(str)

        # Create a scatter matrix (pairplot)
        fig = px.scatter_matrix(
            data,
            dimensions=[col for col in data.columns if col != 'Cluster'],  # All but the Cluster column
            color='Cluster',
            title="Cluster Pairplot",
            opacity=0.6
        )
        fig.update_traces(diagonal_visible=False)  # Hide the diagonal plots
        fig.show(block=True)
