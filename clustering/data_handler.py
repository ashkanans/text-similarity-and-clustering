import os
import shutil

import kagglehub
import pandas as pd


class DataHandler:
    """Handles data downloading and loading."""

    def __init__(self, download_dir="data/raw"):
        self.download_dir = download_dir
        self.data_path = os.path.join(self.download_dir, "housing.csv")

    def download_data(self, force_download=False):
        """Download the dataset to the specified directory."""
        print(f"Downloading dataset to {self.download_dir}...")
        os.makedirs(self.download_dir, exist_ok=True)
        dataset_path = kagglehub.dataset_download(
            "camnugent/california-housing-prices",
            force_download=force_download
        )
        for file_name in os.listdir(dataset_path):
            src_path = os.path.join(dataset_path, file_name)
            dst_path = os.path.join(self.download_dir, file_name)
            if os.path.isfile(src_path):
                shutil.move(src_path, dst_path)
                print(f"Moved {file_name} to {self.download_dir}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Expected dataset file 'housing.csv' not found in {self.download_dir}.")
        print(f"Dataset downloaded to: {self.data_path}")

    def load_data(self):
        """Load the dataset from a CSV file."""
        print(f"Loading data from {self.data_path}...")
        return pd.read_csv(self.data_path)
