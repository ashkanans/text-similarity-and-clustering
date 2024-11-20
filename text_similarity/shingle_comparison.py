import time
import pandas as pd


def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set1 (set): First set of elements.
        set2 (set): Second set of elements.

    Returns:
        float: Jaccard similarity value.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


class ShingleComparison:
    def __init__(self, threshold, shingle_length):
        """
        Initialize the ShingleComparison class.

        Args:
            threshold (float): Jaccard similarity threshold for near-duplicates.
            shingle_length (int): Length of each shingle (k).
        """
        self.elapsed_time = None
        self.df = None
        self.jaccard_values = None
        self.threshold = threshold
        self.shingle_length = shingle_length

    def compare_shingle_sets(self, shingles, descriptions):
        """
        Compare all pairs of shingle sets and identify near-duplicates based on Jaccard similarity.

        Args:
            shingles (list of sets): List of shingle sets.
            descriptions (list of str): Original descriptions corresponding to each set.

        Returns:
            None: Results are stored in the class attributes.
        """
        near_duplicates = []
        jaccard_values = []
        start_time = time.time()

        for i, shingle_set1 in enumerate(shingles):
            for j, shingle_set2 in enumerate(shingles[i + 1:]):
                # Ensure shingle_set1 and shingle_set2 are sets
                if not isinstance(shingle_set1, set):
                    shingle_set1 = set(shingle_set1)
                if not isinstance(shingle_set2, set):
                    shingle_set2 = set(shingle_set2)

                jaccard_sim = jaccard_similarity(shingle_set1, shingle_set2)
                jaccard_values.append(jaccard_sim)
                if jaccard_sim >= self.threshold:
                    description1 = descriptions[i]
                    description2 = descriptions[i + 1 + j]
                    near_duplicates.append((description1, description2, jaccard_sim))

        end_time = time.time()
        self.elapsed_time = end_time - start_time
        self.jaccard_values = jaccard_values

        # Create a DataFrame from the list of near-duplicates
        self.df = pd.DataFrame(near_duplicates, columns=['Description 1', 'Description 2', 'Jaccard Similarity'])
