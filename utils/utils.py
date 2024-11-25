import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2

from text_similarity.shingle_comparison import ShingleComparison
from text_similarity.shingling import Shingling
from text_similarity.text_processing.text_preprocessor import TextPreprocessor

# List of user agents to randomize requests and avoid detection
USER_AGENTS = [
    # Windows User Agents
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:93.0) Gecko/20100101 Firefox/93.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",

    # macOS User Agents
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15",
    # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36",

    # Linux User Agents
    # "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    # "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
    # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
    # "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
]


def get_random_user_agent():
    """
    Selects a random user agent from the predefined list.

    :return: str - A user agent string.
    """
    return random.choice(USER_AGENTS)


def get_headers():
    """
    Generates headers with a random user agent and additional fields.

    :return: dict - HTTP headers for a request.
    """
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.amazon.it/",
    }
    return headers


import datetime
import os


def save_results(args, comparison_results, output_file=None):
    """
    Saves similarity analysis results to a CSV file.

    Args:
        args (argparse.Namespace): The command-line arguments containing configuration settings.
        comparison_results (ShingleComparison): An object containing the results of the similarity analysis.
        output_file (str, optional): The path to save the results file. If None, the path is constructed from args.output.

    Returns:
        None
    """
    # Generate a timestamp for unique file naming
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the output file path
    if not output_file:
        base_output = args.output.rsplit('.', 1)[0]
        output_file = f"{base_output}_{timestamp}.csv"

    print(f"Saving results to {output_file}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the combined DataFrame to CSV
    if isinstance(comparison_results, ShingleComparison):
        comparison_results.df.to_csv(output_file, index=False)
    elif isinstance(comparison_results, pd.DataFrame):
        comparison_results.to_csv(output_file, index=False)

    print("Results successfully saved.")


def generate_shingles(descriptions, k, tokenized):
    """
    Generate shingles from raw or tokenized descriptions.

    Args:
        descriptions (list): List of raw text descriptions or pre-tokenized descriptions.
        k (int): Shingle length.
        tokenized (bool): Whether the input is tokenized or raw.

    Returns:
        list: List of shingle sets for each description.
    """
    shingling = Shingling(documents=descriptions, k=k, is_tokenized=tokenized)
    return shingling.shingles


def preprocess_descriptions(df, tokenize=False, filter_descriptions=False):
    """
        Preprocesses product descriptions from df DataFrame

        :param tokenize: requests preprocessing of data (tokenized) or return raw description text
        :param df: DataFrame
        :param filter_descriptions: (bool) If True, removes duplicate descriptions based on their text.
        :return: list of unique processed descriptions (tokenized lists).
    """
    text_preprocessor = TextPreprocessor(tokenize=tokenize)
    processed_descriptions = []
    unique_descriptions = set()

    if df is not None:
        for description in df['description']:
            processed_tokens = text_preprocessor.preprocess_text(description)
            description_tuple = tuple(processed_tokens)

            if not filter_descriptions or description_tuple not in unique_descriptions:
                processed_descriptions.append(processed_tokens)
                unique_descriptions.add(description_tuple)
    else:
        print("No data to process. Please scrape or load data first.")

    print(f"Processed {len(processed_descriptions)} unique descriptions.")
    return processed_descriptions


def evaluate_methods(naive_df, lsh_df, datasketch_df):
    """
    Evaluates and compares the three methods (Naïve, LSH, DataSketch) for similarity analysis.

    Args:
        naive_df (pd.DataFrame): Baseline DataFrame (Naïve method).
        lsh_df (pd.DataFrame): Results from LSH method.
        datasketch_df (pd.DataFrame): Results from DataSketch method.

    Returns:
        dict: Metrics for LSH and DataSketch methods.
    """
    # Convert to sets of tuples for easier comparison
    naive_pairs = set(naive_df[["Document 1 Index", "Document 2 Index"]].itertuples(index=False, name=None))
    lsh_pairs = set(lsh_df[["Document 1 Index", "Document 2 Index"]].itertuples(index=False, name=None))
    datasketch_pairs = set(datasketch_df[["Document 1 Index", "Document 2 Index"]].itertuples(index=False, name=None))

    def calculate_metrics(method_pairs, baseline_pairs):
        tp = len(method_pairs & baseline_pairs)  # True Positives
        fp = len(method_pairs - baseline_pairs)  # False Positives
        fn = len(baseline_pairs - method_pairs)  # False Negatives
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1-Score": f1_score}

    # Compute metrics
    lsh_metrics = calculate_metrics(lsh_pairs, naive_pairs)
    datasketch_metrics = calculate_metrics(datasketch_pairs, naive_pairs)

    # Return results
    return {"LSH": lsh_metrics, "DataSketch": datasketch_metrics}


def plot_metrics(metrics):
    """
    Plots metrics for LSH and DataSketch methods.

    Args:
        metrics (dict): Metrics dictionary with precision, recall, and F1-score.
    """
    # Extract data
    methods = ["LSH", "DataSketch"]
    precision = [metrics["LSH"]["Precision"], metrics["DataSketch"]["Precision"]]
    recall = [metrics["LSH"]["Recall"], metrics["DataSketch"]["Recall"]]
    f1_score = [metrics["LSH"]["F1-Score"], metrics["DataSketch"]["F1-Score"]]

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    x = range(len(methods))
    plt.bar(x, precision, width=0.2, label="Precision", align='center')
    plt.bar([p + 0.2 for p in x], recall, width=0.2, label="Recall", align='center')
    plt.bar([p + 0.4 for p in x], f1_score, width=0.2, label="F1-Score", align='center')

    plt.xticks([p + 0.2 for p in x], methods)
    plt.ylabel("Score")
    plt.title("Comparison of LSH and DataSketch Metrics")
    plt.legend()
    plt.show(block=True)


def plot_venn_diagram(naive_pairs, method_pairs, method_name):
    """
    Plots a Venn diagram comparing the Naïve method with another method.

    Args:
        naive_pairs (set): Baseline pairs from the Naïve method.
        method_pairs (set): Pairs from the method to compare.
        method_name (str): Name of the method (e.g., "LSH").
    """
    plt.figure(figsize=(8, 8))
    venn2(
        [naive_pairs, method_pairs],
        ("Naïve", method_name)
    )
    plt.title(f"Venn Diagram: Naïve (brute-force) vs {method_name}")
    plt.show(block=True)





def plot_similarity_heatmap(df, title="Jaccard Similarity Heatmap"):
    """
    Plots a heatmap of Jaccard similarities between document pairs.

    Args:
        df (pd.DataFrame): DataFrame with columns ['Document 1 Index', 'Document 2 Index', 'Jaccard Similarity'].
        title (str): Title for the heatmap.
    """
    # Pivot the DataFrame to create a similarity matrix
    similarity_matrix = df.pivot(index="Document 1 Index", columns="Document 2 Index", values="Jaccard Similarity")

    # Fill NaNs with 0 for better visualization
    similarity_matrix = similarity_matrix.fillna(0)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="coolwarm", annot=False)
    plt.title(title)
    plt.xlabel("Document 2 Index")
    plt.ylabel("Document 1 Index")
    plt.show(block=True)


def plot_heatmap(naive_df, lsh_df, datasketch_df):
    plot_similarity_heatmap(naive_df, title="Naïve (brute-force) Jaccard Similarity Heatmap")
    plot_similarity_heatmap(lsh_df, title="LSH Jaccard Similarity Heatmap")
    plot_similarity_heatmap(datasketch_df, title="DataSketch Jaccard Similarity Heatmap")


def plot_cumulative_distribution(df_list, labels, title="Cumulative Distribution of Jaccard Similarities"):
    """
    Plots cumulative distributions of Jaccard similarities for multiple methods.

    Args:
        df_list (list of pd.DataFrame): List of DataFrames to compare.
        labels (list of str): Labels for each DataFrame.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))

    for df, label in zip(df_list, labels):
        # Sort Jaccard similarities
        similarities = df["Jaccard Similarity"].sort_values()
        cumulative = [i / len(similarities) for i in range(len(similarities))]
        plt.plot(similarities, cumulative, label=label)

    plt.title(title)
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Cumulative Percentage")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)


def plot_precision_recall(metrics):
    """
    Plots precision, recall, and F1-score for LSH and DataSketch methods.

    Args:
        metrics (dict): Dictionary containing metrics for LSH and DataSketch methods.
    """
    methods = ["LSH", "DataSketch"]
    precision = [metrics["LSH"]["Precision"], metrics["DataSketch"]["Precision"]]
    recall = [metrics["LSH"]["Recall"], metrics["DataSketch"]["Recall"]]
    f1_score = [metrics["LSH"]["F1-Score"], metrics["DataSketch"]["F1-Score"]]

    # Plot grouped bar chart
    x = range(len(methods))
    plt.figure(figsize=(10, 6))
    plt.bar(x, precision, width=0.2, label="Precision", align='center')
    plt.bar([p + 0.2 for p in x], recall, width=0.2, label="Recall", align='center')
    plt.bar([p + 0.4 for p in x], f1_score, width=0.2, label="F1-Score", align='center')

    plt.xticks([p + 0.2 for p in x], methods)
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1-Score Comparison")
    plt.legend()
    plt.show(block=True)


def plot_similarity_boxplot(df_list, labels, title="Jaccard Similarity Box Plot"):
    """
    Plots a box plot comparing Jaccard similarity distributions for multiple methods.

    Args:
        df_list (list of pd.DataFrame): List of DataFrames to compare.
        labels (list of str): Labels for each DataFrame.
        title (str): Title for the plot.
    """
    similarities = [df["Jaccard Similarity"] for df in df_list]
    plt.figure(figsize=(10, 6))
    plt.boxplot(similarities, labels=labels, patch_artist=True)
    plt.title(title)
    plt.ylabel("Jaccard Similarity")
    plt.grid(True)
    plt.show(block=True)


def plot_similarity_distribution(df_list, labels, bins=20, title="Jaccard Similarity Distribution"):
    """
    Plots a histogram comparing Jaccard similarity distributions for multiple methods.

    Args:
        df_list (list of pd.DataFrame): List of DataFrames to compare.
        labels (list of str): Labels for each DataFrame.
        bins (int): Number of bins for the histogram.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 6))

    for df, label in zip(df_list, labels):
        plt.hist(df["Jaccard Similarity"], bins=bins, alpha=0.5, label=label)

    plt.title(title)
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show(block=True)
