import os
import pandas as pd

def save_to_tsv(data, file_path):
    """
    Saves a list of dictionaries to a TSV file.

    :param data: list of dict - Data to be saved.
    :param file_path: str - Path to the TSV file.
    """
    if not data:
        print("No data to save.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Remove duplicates based on product description and save to TSV
    df.drop_duplicates(subset=['description', 'price', 'url', 'star_rating', 'reviews'], inplace=True)
    df.to_csv(file_path, sep='\t', index=False)

    print(f"Data saved to {file_path}")

def load_dataset(file_path):
    """
    Loads a TSV file into a pandas DataFrame.

    :param file_path: str - Path to the TSV file.
    :return: pandas.DataFrame - Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the TSV file into a DataFrame and drop missing values
    df = pd.read_csv(file_path, sep='\t')
    # df.dropna(inplace=True)

    print(f"Data loaded from {file_path}")
    return df

def preprocess_column(df, column_name):
    """
    Preprocesses a specific column in the DataFrame (e.g., removing special characters or normalizing text).

    :param df: pandas.DataFrame - Input DataFrame.
    :param column_name: str - Column to preprocess.
    :return: pandas.DataFrame - DataFrame with the preprocessed column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Example: Strip whitespace and convert to lowercase
    df[column_name] = df[column_name].str.strip().str.lower()
    print(f"Preprocessed column: {column_name}")
    return df
