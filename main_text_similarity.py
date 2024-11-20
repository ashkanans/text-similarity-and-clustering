import argparse
import os

import numpy as np
import pandas as pd

from text_similarity.shingling import Shingling
from text_similarity.minwise_hash_signatures import MinwiseHashSignature
from text_similarity.lsh import LSH
from text_similarity.shingle_comparison import ShingleComparison


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run text similarity analysis using shingles, MinHash, and LSH.")
    parser.add_argument("--scrape", action="store_true", help="Scrape Amazon data based on keywords.")
    parser.add_argument("--keyword", type=str, default="computer, pc, portatile, laptop",
                        help="Keywords for Amazon search, separated by commas.")
    parser.add_argument("--num_pages", type=int, default=5, help="Number of pages to scrape.")
    parser.add_argument("--path", type=str, default=os.path.join('data', 'raw', 'computer_results_default.tsv'),
                        help="TSV file path for Amazon products.")
    parser.add_argument("--k", type=int, default=10, help="Shingle length.")
    parser.add_argument("--threshold", type=float, default=0.80, help="Jaccard similarity threshold.")
    parser.add_argument("--num_hashes", type=int, default=100, help="Number of hash functions for MinHash.")
    parser.add_argument("--output", type=str, default=os.path.join('data', 'processed', 'near_duplicates.csv'),
                        help="Output file path for near duplicates.")

    args = parser.parse_args()
    return args


def load_or_scrape_data(args):
    from text_similarity.scraping.AmazonScraper import AmazonScraper
    amazon_scraper = AmazonScraper(args.keyword, args.num_pages)
    if args.scrape:
        print("Scraping Amazon products...")
        amazon_scraper.scrape_amazon_products()
        amazon_scraper.save_to_tsv()
        amazon_scraper.load_dataset(amazon_scraper.scraped_results)
    else:
        if os.path.exists(args.path):
            print(f"Loading Amazon data from {args.path}...")
            amazon_scraper.load_dataset(args.path)
        else:
            raise FileNotFoundError(f"Dataset file not found at {args.path}")

    processed_descriptions = amazon_scraper.preprocess_descriptions()
    return processed_descriptions


def generate_shingles(processed_descriptions, k):
    shingles_list = [Shingling(desc, k).shingles for desc in processed_descriptions]
    return shingles_list


def perform_similarity_analysis(shingles_list, descriptions, args):
    print("Generating MinHash signatures...")
    minwise = MinwiseHashSignature(num_hashes=args.num_hashes, num_elements=len(shingles_list), signature_length=args.k)
    signatures = minwise.generate_signatures(shingles_list)

    print("Performing parameter tuning with S-curve analysis...")
    # Initialize LSH
    shingle_comparison = ShingleComparison(threshold=args.threshold, shingle_length=args.k)
    lsh = LSH(minwise_hash_signature=minwise, shingle_comparison=shingle_comparison, num_buckets=10)

    # Generate a range of Jaccard similarities (0 to 1) for analysis
    jaccard_values = np.linspace(0, 1, 100)

    # Perform S-curve analysis to find optimal r and b values
    lsh.s_curve_plot_and_analysis(jaccard_values=jaccard_values, max_r=50, max_b=50, threshold=args.threshold)
    print(f"Optimal parameters: r={lsh.r}, b={lsh.b}, threshold_prob={lsh.threshold_prob:.2f}")

    print("Performing Locality Sensitive Hashing (LSH)...")
    lsh.set_r_b_values(r=lsh.r, b=lsh.b, threshold_prob=args.threshold)
    lsh.find_near_duplicates(shingles_list)

    print("Comparing all pairs of shingles for near duplicates...")
    shingle_comparison.compare_shingle_sets(shingles_list, descriptions)

    print(f"Saving results to {args.output}...")
    shingle_comparison.df.to_csv(args.output, index=False)
    print(f"Elapsed time for pairwise comparison: {shingle_comparison.elapsed_time:.2f} seconds")
    print(f"Elapsed time for LSH: {lsh.elapsed_time_lsh:.2f} seconds")



def main():
    args = parse_arguments()
    processed_descriptions = load_or_scrape_data(args)
    shingles_list = generate_shingles(processed_descriptions, args.k)
    perform_similarity_analysis(shingles_list, processed_descriptions, args)


if __name__ == "__main__":
    main()
