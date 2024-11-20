# A simple Spark app to test if it works properly!
import os

from pyspark.sql import SparkSession
from pyspark.sql import Row

try:
    # Specify the Python executable from your virtual environment
    venv_python_path = "../.venv/Scripts/python.exe"

    os.environ["PYSPARK_PYTHON"] = venv_python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = venv_python_path

    # Configure Spark to use the virtual environment's Python
    spark = SparkSession.builder \
        .appName("SparkTest") \
        .master("local[*]") \
        .config("spark.pyspark.python", venv_python_path) \
        .config("spark.pyspark.driver.python", venv_python_path) \
        .getOrCreate()

    test_data = [Row(id=1, name="Alice"), Row(id=2, name="Bob"), Row(id=3, name="Charlie")]
    df = spark.createDataFrame(test_data)

    df.show()

    row_count = df.count()
    print(f"Row count: {row_count}")

    spark.stop()

    print("Spark test completed successfully.")

except Exception as e:
    print("Spark test failed with exception:", e)
