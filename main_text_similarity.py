import argparse
import os

import numpy as np

from text_similarity.scraping.amazon_scraper import AmazonScraper
from text_similarity.scraping.data_processing import load_dataset
from text_similarity.scraping.visualization import plot_token_length_distribution

from text_similarity.minwise_hash_signatures import MinwiseHashSignature
from text_similarity.lsh import LSH
from text_similarity.shingle_comparison import ShingleComparison

from utils.utils import save_results, generate_shingles, preprocess_descriptions


def parse_arguments():
    """
    Parses command-line arguments for the application.
    """
    parser = argparse.ArgumentParser(description="Run text similarity analysis using shingles, MinHash, and LSH.")
    parser.add_argument("--scrape", action="store_true", help="Scrape Amazon data based on keywords.")
    parser.add_argument("--keyword", type=str, default="computer, pc, portatile, laptop",
                        help="Keywords for Amazon search, separated by commas.")
    parser.add_argument("--num_pages", type=int, default=10, help="Number of pages to scrape.")
    parser.add_argument("--path", type=str, default=os.path.join('data', 'raw', 'computer_results_2024-11-22.tsv'),
                        help="TSV file path for Amazon products.")
    parser.add_argument("--k", type=int, default=10, help="Shingle length.")
    parser.add_argument("--threshold", type=float, default=0.80, help="Jaccard similarity threshold.")
    parser.add_argument("--num_hashes", type=int, default=100, help="Number of hash functions for MinHash.")
    parser.add_argument("--output", type=str, default=os.path.join('data', 'processed', 'near_duplicates.csv'),
                        help="Output file path for near duplicates.")
    parser.add_argument("--tokenize", action="store_true", default=True,
                        help="Flag to tokenize descriptions and return tokenized data.")
    parser.add_argument("--max_r", type=int, default=50, help="Maximum number of rows for LSH S-curve analysis.")
    parser.add_argument("--max_b", type=int, default=50, help="Maximum number of bands for LSH S-curve analysis.")

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


def perform_similarity_analysis(shingles_list, df, args):
    """
    Performs similarity analysis using MinHash and LSH.
    """
    print("Generating MinHash signatures...")
    minwise = MinwiseHashSignature(num_hashes=args.num_hashes, num_elements=len(shingles_list), signature_length=args.k)

    print("Performing parameter tuning with S-curve analysis...")
    shingle_comparison = ShingleComparison(threshold=args.threshold, shingle_length=args.k)
    lsh = LSH(minwise_hash_signature=minwise, shingle_comparison=shingle_comparison, num_buckets=10)

    jaccard_values = np.linspace(0, 1, 100)
    lsh.s_curve_plot_and_analysis(jaccard_values=jaccard_values, max_r=args.max_r, max_b=args.max_b,
                                  threshold=args.threshold)
    print(f"Optimal parameters: r={lsh.r}, b={lsh.b}, threshold_prob={lsh.threshold_prob:.2f}")

    print("Performing Locality Sensitive Hashing (LSH)...")
    lsh.set_r_b_values(r=lsh.r, b=lsh.b, threshold_prob=args.threshold)
    lsh.find_near_duplicates(shingles_list)

    print("Comparing all pairs of shingles for near duplicates...")
    shingle_comparison.compare_shingle_sets(shingles_list, df['Description'].tolist())
    save_results(args, shingle_comparison)


def main():
    args = parse_arguments()
    df = load_or_scrape_data(args)

    print("Preprocessing data...")
    processed_data = preprocess_descriptions(df, args.tokenize)

    print("Generating shingles...")
    shingles_list = generate_shingles(processed_data, args.k, args.tokenize)

    print("Performing similarity analysis...")
    perform_similarity_analysis(shingles_list, processed_data, args)


if __name__ == "__main__":
    main()
