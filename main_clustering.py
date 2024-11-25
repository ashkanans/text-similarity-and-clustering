import argparse

from clustering.california_housing_runner import CaliforniaHousingRunner
from clustering.clustering import Clustering
from clustering.data_handler import DataHandler
from clustering.feature_engineering import FeatureEngineer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="California Housing Clustering")
    parser.add_argument("actions", nargs="+", type=str,
                        help="List of actions to perform (e.g., download load elbow_raw clustering_raw feature_engineering elbow_engineered clustering_engineered).")
    parser.add_argument("--data_path", type=str, default="data/raw/housing.csv",
                        help="Path to the data file.")
    args = parser.parse_args()

    # Dependency injection
    data_handler = DataHandler(download_dir="data/raw")
    feature_engineer = FeatureEngineer()
    clustering = Clustering()

    runner = CaliforniaHousingRunner(data_handler, feature_engineer, clustering)
    runner.run(args.actions)
