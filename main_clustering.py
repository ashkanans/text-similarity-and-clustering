import argparse

from clustering.california_housing_runner import CaliforniaHousingRunner
from clustering.clustering import Clustering
from clustering.data_handler import DataHandler
from clustering.feature_engineering import FeatureEngineer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="California Housing Clustering")
    parser.add_argument(
        "actions", nargs="+", type=str,
        help="List of actions to perform (e.g., download, load, clustering_raw). "
             "For example: python main_clustering.py load clustering_raw"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/raw/housing.csv",
        help="Path to the dataset file (default: data/raw/housing.csv)."
    )
    parser.add_argument(
        "--score_method", type=str, default="silhouette",
        choices=["silhouette", "elbow", "davies_bouldin", "calinski_harabasz"],
        help="Choose the scoring method: 'silhouette', 'elbow', 'davies_bouldin', or 'calinski_harabasz'. "
             "Default is 'silhouette'."
    )
    args = parser.parse_args()

    # Dependency injection
    data_handler = DataHandler(download_dir="data/raw")
    feature_engineer = FeatureEngineer()
    clustering = Clustering()

    runner = CaliforniaHousingRunner(data_handler, feature_engineer, clustering, score_method=args.score_method)
    runner.run(args.actions)
