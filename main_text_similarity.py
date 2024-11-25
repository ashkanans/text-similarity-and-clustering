import argparse
import os
import time

import numpy as np
import pandas as pd

from text_similarity.lsh import LSH
from text_similarity.minwise_hash_signatures import MinwiseHashSignature
from text_similarity.scraping.amazon_scraper import AmazonScraper
from text_similarity.scraping.data_processing import load_dataset
from text_similarity.shingle_comparison import ShingleComparison
from text_similarity.text_processing.data_sketch_lsh import DataSketchLSH
from utils.utils import save_results, generate_shingles, preprocess_descriptions, evaluate_methods, plot_metrics, \
    plot_venn_diagram, plot_heatmap, plot_cumulative_distribution, plot_precision_recall, plot_similarity_boxplot, \
    plot_similarity_distribution


def parse_arguments():
    """
    Parses command-line arguments for the application.
    """
    parser = argparse.ArgumentParser(description="Run text similarity analysis using shingles, MinHash, and LSH.")
    parser.add_argument("--scrape", action="store_true", help="Scrape Amazon data based on keywords.")
    parser.add_argument("--keyword", type=str, default="computer, pc, portatile, laptop",
                        help="Keywords for Amazon search, separated by commas.")
    parser.add_argument("--num_pages", type=int, default=10, help="Number of pages to scrape.")
    parser.add_argument("--path", type=str, default=os.path.join('data', 'raw', 'laptops_results_2024-11-23.tsv'),
                        help="TSV file path for Amazon products.")
    parser.add_argument("--k", type=int, default=10, help="Shingle length.")
    parser.add_argument("--threshold", type=float, default=0.80, help="Jaccard similarity threshold.")
    parser.add_argument("--output", type=str, default=os.path.join('data', 'processed', 'near_duplicates.csv'),
                        help="Output file path for near duplicates.")
    parser.add_argument("--tokenize", action="store_true", default=False,
                        help="Flag to tokenize descriptions and return tokenized data.")
    parser.add_argument("--max_r", type=int, default=20, help="Maximum number of rows for LSH S-curve analysis.")
    parser.add_argument("--max_b", type=int, default=20, help="Maximum number of bands for LSH S-curve analysis.")
    parser.add_argument("--plot", type=int, default=False, help="Plot the comparison charts for the methods")

    args = parser.parse_args()
    return args


def load_or_scrape_data(args):
    """
    Loads or scrapes Amazon data based on the provided arguments.
    """
    amazon_scraper = AmazonScraper(args.keyword, args.num_pages)
    if args.scrape:
        print("Scraping Amazon products...")
        amazon_scraper.scrape_amazon_products()
        amazon_scraper.save_results()
        data_path = amazon_scraper.scraped_results
    else:
        if not os.path.exists(args.path):
            raise FileNotFoundError(f"Dataset file not found at {args.path}")
        data_path = args.path

    print(f"Loading data from {data_path}...")
    df = load_dataset(data_path)
    return df


def dict_to_dataframe(input_dict):
    """
    Converts a dictionary or defaultdict into a pandas DataFrame
    with columns: Document 1 Index, Document 2 Index, Jaccard Similarity.

    Args:
        input_dict (dict or defaultdict): The input dictionary-like object.

    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - "Document 1 Index"
                      - "Document 2 Index"
                      - "Jaccard Similarity"
    """
    rows = []
    for doc1, matches in input_dict.items():
        for doc2, similarity in matches:
            rows.append({
                "Document 1 Index": doc1,
                "Document 2 Index": doc2,
                "Jaccard Similarity": similarity
            })

    return pd.DataFrame(rows)


def perform_lsh_similarity_analysis(shingles_list, args):
    """
    Performs similarity analysis using MinHash and Locality-Sensitive Hashing (LSH).

    Args:
        shingles_list (list of sets): A list of shingle sets for all documents.
        args (argparse.Namespace): Command-line arguments including configuration settings.

    Returns:
        None
    """
    start_time = time.time()

    print("Performing parameter tuning with S-curve analysis...")
    lsh = LSH(num_buckets=100)

    # Optimize (r, b) parameters with S-curve analysis
    jaccard_values = np.linspace(0, 1, 100)
    lsh.s_curve_plot_and_analysis(
        jaccard_values=jaccard_values,
        max_r=args.max_r,
        max_b=args.max_b,
        threshold=args.threshold
    )
    print(f"Optimal parameters: r={lsh.r}, b={lsh.b}")

    # Perform LSH
    print("Performing Locality Sensitive Hashing (LSH)...")
    lsh.set_r_b_values(r=lsh.r, b=lsh.b)
    # fix r and b to a low value for testing
    # lsh.set_r_b_values(r=5, b=5)

    # Generate MinHash signatures
    print("Generating MinHash signatures...")
    num_hashes = lsh.r * lsh.b
    minwise_hash_signature = MinwiseHashSignature(num_hashes=num_hashes, num_elements=len(shingles_list))
    signatures = minwise_hash_signature.generate_signatures(shingles_list)

    # Index signatures in hash tables
    print("Indexing MinHash signatures...")
    lsh.index_signatures(signatures)

    # Find near-duplicates
    print("Finding near-duplicates...")
    candidates_dict = lsh.find_near_duplicates(signatures, threshold=args.threshold)

    candidates_df = dict_to_dataframe(candidates_dict)

    output_file = os.path.join(os.path.dirname(args.output), "near_duplicates_lsh.csv")
    save_results(args, candidates_df, output_file)

    elapsed_time = time.time() - start_time
    print(f"LSH similarity analysis completed in {elapsed_time:.2f} seconds.")
    print(f"Number of near duplicates found using LSH: {len(candidates_df)}")
    print(f"Results saved to: {output_file}")
    return candidates_df


def perform_naive_similarity_analysis(shingles_list, processed_data, args):
    """
    Performs similarity analysis by comparing all pairs of shingles (brute force).

    Args:
        shingles_list (list of sets): A list of shingle sets for all documents.
        processed_data (list): A list where each element is a product description.
                               Tokenized (list of words) if args.tokenize is True, raw strings otherwise.
        args (argparse.Namespace): Command-line arguments including configuration settings.

    Returns:
        None
    """
    start_time = time.time()

    descriptions = [" ".join(desc) for desc in processed_data] if args.tokenize else processed_data

    shingle_comparison = ShingleComparison(threshold=args.threshold, shingle_length=args.k)
    shingle_comparison.compare_shingle_sets(shingles_list, descriptions)

    output_file = os.path.join(os.path.dirname(args.output), "near_duplicates_naive.csv")
    save_results(args, shingle_comparison, output_file)

    elapsed_time = time.time() - start_time
    num_duplicates = len(shingle_comparison.df)
    print(f"Naïve similarity analysis completed in {elapsed_time:.2f} seconds.")
    print(f"Number of near duplicates found using Naïve method: {num_duplicates}")
    print(f"Results saved to: {output_file}")
    return shingle_comparison.df


def perform_datasketch_similarity_analysis(shingles_list, args):
    """
    Performs similarity analysis using the DataSketch library for MinHash and LSH.

    Args:
        shingles_list (list of sets): A list of shingle sets for all documents.
        args (argparse.Namespace): Command-line arguments including configuration settings.

    Returns:
        None
    """
    start_time = time.time()

    lsh = DataSketchLSH(threshold=args.threshold, num_perm=320)
    lsh.add_shingles_list(shingles_list)

    output_file = os.path.join(os.path.dirname(args.output), "near_duplicates_data_sketch.csv")
    num_duplicates, results_df = lsh.find_near_duplicates(args)

    elapsed_time = time.time() - start_time
    print(f"DataSketch similarity analysis completed in {elapsed_time:.2f} seconds.")
    print(f"Number of near duplicates found using DataSketch: {num_duplicates}")
    print(f"Results saved to: {output_file}")
    return results_df


def main():
    args = parse_arguments()
    df = load_or_scrape_data(args)

    print("Preprocessing data...")
    processed_data = preprocess_descriptions(df, args.tokenize)

    print("Generating shingles...")
    shingles_list = generate_shingles(processed_data, args.k, args.tokenize)

    print("\nStep 1. Performing similarity analysis using LSH technique...")
    lsh_df = perform_lsh_similarity_analysis(shingles_list, args)

    print("\nStep 2. Performing similarity analysis using Naïve (brute-force) technique ...")
    naive_df = perform_naive_similarity_analysis(shingles_list, processed_data, args)

    print("\nStep 3. Performing similarity analysis using DataSketch...")
    datasketch_df = perform_datasketch_similarity_analysis(shingles_list, args)

    if args.plot:
        # Evaluate methods
        metrics = evaluate_methods(naive_df, lsh_df, datasketch_df)

        # Plot metrics
        plot_metrics(metrics)

        # Plot Venn diagrams
        naive_pairs = set(naive_df[["Document 1 Index", "Document 2 Index"]].itertuples(index=False, name=None))
        lsh_pairs = set(lsh_df[["Document 1 Index", "Document 2 Index"]].itertuples(index=False, name=None))
        datasketch_pairs = set(
            datasketch_df[["Document 1 Index", "Document 2 Index"]].itertuples(index=False, name=None))

        plot_venn_diagram(naive_pairs, lsh_pairs, "LSH")
        plot_venn_diagram(naive_pairs, datasketch_pairs, "DataSketch")

        plot_heatmap(datasketch_df=datasketch_df, naive_df=naive_df, lsh_df=lsh_df)

        plot_cumulative_distribution(
            [naive_df, lsh_df, datasketch_df],
            labels=["Naïve (brute-force)", "LSH", "DataSketch"],
            title="Cumulative Distribution of Jaccard Similarities"
        )

        plot_precision_recall(metrics)

        plot_similarity_distribution(
            [naive_df, lsh_df, datasketch_df],
            labels=["Naïve (brute-force)", "LSH", "DataSketch"],
            bins=20,
            title="Jaccard Similarity Distribution Across Methods"
        )

        plot_similarity_boxplot(
            [naive_df, lsh_df, datasketch_df],
            labels=["Naïve (brute-force)", "LSH", "DataSketch"],
            title="Jaccard Similarity Box Plot"
        )


if __name__ == "__main__":
    main()
