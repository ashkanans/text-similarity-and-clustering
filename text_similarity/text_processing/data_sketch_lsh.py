import os

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


class DataSketchLSH:
    def __init__(self, threshold=0.8, num_perm=128):
        """
        Initializes the DataSketchLSH class for near-duplicate detection.

        Args:
            threshold (float): The Jaccard similarity threshold for near-duplicates.
            num_perm (int): Number of permutations (hash functions) for Min-Hashing.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.minhashes = []
        self.shingles_list = []

    def add_shingles_list(self, shingles_list):
        """
        Adds shingles (sets) to the LSH index.

        Args:
            shingles_list (list of sets): List of sets, where each set represents the shingles of a document.
        """
        self.shingles_list = shingles_list
        with tqdm(total=len(shingles_list), desc="Adding shingles to LSH") as progress_bar:
            for i, shingles in enumerate(shingles_list):
                minhash = MinHash(num_perm=self.num_perm)
                for shingle in shingles:
                    minhash.update(shingle.encode('utf-8'))
                self.minhashes.append(minhash)
                self.lsh.insert(f"doc_{i}", minhash)
                progress_bar.update(1)

    def find_near_duplicates(self, args):
        """
        Finds near-duplicate documents and writes them to a CSV file with their Jaccard similarity.

        Args:
            args (argparse.Namespace): Command-line arguments containing output file path and other settings.

        Returns:
            int: The number of filtered near-duplicate pairs.
            pd.DataFrame: A DataFrame containing the filtered near-duplicate pairs.
        """
        results = []

        for i, doc_minhash in enumerate(self.minhashes):
            candidates = self.lsh.query(doc_minhash)
            for candidate in candidates:
                if candidate == f"doc_{i}":
                    continue

                candidate_index = int(candidate.split("_")[1])
                jaccard_sim = doc_minhash.jaccard(self.minhashes[candidate_index])

                # Save result as (Document 1 Index, Document 2 Index, Jaccard Similarity)
                results.append(
                    {"Document 1 Index": i, "Document 2 Index": candidate_index, "Jaccard Similarity": jaccard_sim})

        results_df = pd.DataFrame(results)

        filtered_results_df = results_df.loc[results_df["Jaccard Similarity"] >= args.threshold]

        output_file = os.path.join(os.path.dirname(args.output), "near_duplicates_data_sketch.csv")

        output_file_with_timestamp = f"{output_file.rsplit('.', 1)[0]}.csv"
        filtered_results_df.to_csv(output_file_with_timestamp, index=False)

        return len(filtered_results_df), filtered_results_df
