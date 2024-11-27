import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class FeatureEngineer:
    """Handles feature engineering operations."""

    def preprocess_data(self, data,
                        scale_numeric=True,
                        one_hot_encode=True,
                        add_interaction_features=True,
                        log_transform_features=True,
                        discretize_age=True,
                        handle_missing=True):
        """
        Apply various feature engineering techniques based on flags.

        Parameters:
        - scale_numeric: Scale numeric features.
        - one_hot_encode: Apply one-hot encoding to categorical features.
        - add_interaction_features: Add interaction features (e.g., rooms per household).
        - log_transform_features: Apply log transformation to specified skewed features.
        - discretize_age: Discretize 'housing_median_age' into bins.
        - handle_missing: Handle missing data via imputation.

        Returns:
        - Engineered DataFrame
        """
        data = data.copy()

        numeric_features = ['longitude', 'latitude', 'total_rooms',
                            'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
        categorical_features = ['ocean_proximity']

        # Handle missing data
        if handle_missing:
            imputer = SimpleImputer(strategy="median")
            data[numeric_features] = imputer.fit_transform(data[numeric_features])
            print("Missing data imputed.")

        # Scale numeric features
        if scale_numeric:
            scaler = StandardScaler()
            data[numeric_features] = scaler.fit_transform(data[numeric_features])
            print("Numeric features scaled.")

        # One-hot encode categorical features
        if one_hot_encode:
            encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first for redundancy
            encoded = pd.DataFrame(encoder.fit_transform(data[categorical_features]),
                                   columns=encoder.get_feature_names_out(categorical_features),
                                   index=data.index)
            data = pd.concat([data.drop(columns=categorical_features), encoded], axis=1)
            print("Categorical features one-hot encoded.")

        # Add interaction features
        if add_interaction_features:
            data['rooms_per_household'] = data['total_rooms'] / data['households']
            data['bedrooms_per_household'] = data['total_bedrooms'] / data['households']
            print("Interaction features added.")

        # Log transformation
        if log_transform_features:
            skewed_features = ['population', 'median_income']
            for feature in skewed_features:
                if feature in data:
                    data[f'log_{feature}'] = np.log1p(data[feature])
            print("Log transformation applied.")

        # Discretize and one-hot encode 'housing_median_age'
        if discretize_age:
            bins = [0, 20, 40, np.inf]
            labels = ['0-20', '20-40', '40+']
            data['age_bin'] = pd.cut(data['housing_median_age'], bins=bins, labels=labels)

            # Apply one-hot encoding
            age_bin_encoded = pd.get_dummies(data['age_bin'], prefix='age_bin', drop_first=False)
            data = pd.concat([data.drop(columns=['housing_median_age', 'age_bin']), age_bin_encoded], axis=1)
            print("'housing_median_age' discretized and one-hot encoded.")

        print("Feature engineering completed.")
        return data
