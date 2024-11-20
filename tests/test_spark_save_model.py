import os
import pickle

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Ensure the output directory exists
os.makedirs("models", exist_ok=True)


def main():
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("LogisticRegressionExample") \
        .getOrCreate()

    # Create a simple DataFrame
    data = spark.createDataFrame([
        (0, 0.0, 1.1, 1.0),
        (1, 2.0, 1.0, 0.0),
        (2, 2.0, 1.3, 1.0),
        (3, 0.0, 1.2, 0.0),
        (4, 0.0, -0.5, 1.0)
    ], ["id", "feature1", "feature2", "label"])

    # Prepare features
    assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")
    assembled_data = assembler.transform(data).select("id", "features", "label")

    # Train Logistic Regression Model
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(assembled_data)

    # Save model coefficients and intercept using pickle
    model_data = {
        "coefficients": lr_model.coefficients.toArray(),
        "intercept": lr_model.intercept
    }
    model_path = "models/logistic_regression.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model parameters saved successfully at: {model_path}")

    # Load the model parameters using pickle
    with open(model_path, "rb") as f:
        loaded_model_data = pickle.load(f)
    print("Model parameters loaded successfully.")

    # Print the loaded parameters
    print("Coefficients:", loaded_model_data["coefficients"])
    print("Intercept:", loaded_model_data["intercept"])

    spark.stop()


if __name__ == "__main__":
    main()
