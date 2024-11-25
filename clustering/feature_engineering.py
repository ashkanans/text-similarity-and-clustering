import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureEngineer:
    """Handles feature engineering operations."""

    @staticmethod
    def preprocess_data(data):
        """Apply feature engineering: scaling and one-hot encoding."""
        numeric_features = data.select_dtypes(include=[np.number]).columns
        categorical_features = ['ocean_proximity']

        # Scale numeric features
        scaler = StandardScaler()
        data[numeric_features] = scaler.fit_transform(data[numeric_features])

        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name
        encoded = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
        engineered_data = pd.concat([data.drop(columns=categorical_features), encoded], axis=1)
        print("Feature engineering completed.")
        return engineered_data
