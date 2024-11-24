import time
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go


class LSH:
    def __init__(self, num_buckets):
        self.num_buckets = num_buckets
        self.hash_tables = [defaultdict(list) for _ in range(num_buckets)]
        self.b = None  # Number of bands
        self.r = None  # Rows per band

    def _split_into_bands(self, signatures):
        """
        Splits Min-Hash signatures into bands for indexing and querying.
        """
        num_hashes, num_signatures = signatures.shape
        if self.b * self.r != num_hashes:
            raise ValueError(f"Number of hashes ({num_hashes}) must equal b * r (b={self.b}, r={self.r}).")

        bands = np.array_split(signatures, self.b, axis=0)
        return bands

    def index_signatures(self, signatures):
        """
        Indexes the Min-Hash signatures into hash tables based on bands.
        """
        start_time = time.time()
        bands = self._split_into_bands(signatures)

        for band_idx, band in enumerate(bands):
            for doc_idx, band_signature in enumerate(band.T):  # Transpose for per-document
                bucket = tuple(band_signature)
                self.hash_tables[band_idx][bucket].append(doc_idx)

        elapsed_time = time.time() - start_time
        print(f"Indexing completed in {elapsed_time:.2f} seconds.")

    def query_signature(self, query_signature):
        """
        Queries the hash tables for candidates of a given query signature.
        """
        bands = self._split_into_bands(query_signature[:, None])  # Add axis for consistency

        candidate_set = set()
        for band_idx, band_signature in enumerate(bands):
            bucket = tuple(band_signature.flatten())
            candidate_set.update(self.hash_tables[band_idx].get(bucket, []))

        return candidate_set

    def find_near_duplicates(self, signatures, threshold=0.8):
        """
        Finds near-duplicates for all signatures.
        """
        start_time = time.time()
        candidates = defaultdict(list)

        for query_idx, query_signature in enumerate(signatures.T):
            candidate_indices = self.query_signature(query_signature)

            for candidate_idx in candidate_indices:
                if query_idx != candidate_idx:  # Avoid self-comparison
                    # Calculate Jaccard similarity
                    jaccard_sim = self._calculate_jaccard_similarity(
                        signatures[:, query_idx], signatures[:, candidate_idx]
                    )
                    if jaccard_sim >= threshold:
                        candidates[query_idx].append((candidate_idx, jaccard_sim))

        elapsed_time = time.time() - start_time
        print(f"Near-duplicate detection completed in {elapsed_time:.2f} seconds.")
        return candidates

    @staticmethod
    def _calculate_jaccard_similarity(signature1, signature2):
        """
        Calculates Jaccard similarity between two Min-Hash signatures.
        """
        return np.sum(signature1 == signature2) / len(signature1)

    def set_r_b_values(self, r, b):
        """
        Sets the number of rows per band (r) and the number of bands (b).
        """
        self.r = r
        self.b = b

    def s_curve_plot_and_analysis(self, jaccard_values, max_r, max_b, threshold=0.8):
        """
        Plot S-curves showing the first, last, the chosen and all (r, b) combinations.
        The last optimized combination is highlighted in red, while others are blue.

        :param jaccard_values: List of Jaccard similarity values.
        :param max_r: Maximum number of rows (r).
        :param max_b: Maximum number of bands (b).
        :param threshold: Jaccard similarity threshold.
        """
        fig = go.Figure()

        self.jaccard_values = sorted(jaccard_values)
        self.r_list = []
        self.b_list = []
        self.threshold_prob_list = []
        max_slope_at_threshold = float('-inf')
        last_r, last_b = None, None  # To store the last optimized (r, b)

        def is_step_shape(prob, index_th, epsilon=0.01, slope_threshold=0.1):
            # Calculate the slope at the beginning and end
            start_slope = (prob[1] - prob[0]) / epsilon
            end_slope = (prob[-1] - prob[-2]) / epsilon
            threshold_slope = (prob[index_th + 1] - prob[index_th - 1]) / (2 * epsilon)

            # Check conditions for step shape
            is_small_slope_at_start = abs(start_slope) < slope_threshold
            is_small_slope_at_end = abs(end_slope) < slope_threshold

            return is_small_slope_at_start, is_small_slope_at_end, threshold_slope

        for r in range(1, max_r + 1):
            for b in range(1, max_b + 1):
                probabilities = 1 - np.power((1 - np.power(self.jaccard_values, r)), b)
                index = int(threshold * len(probabilities))

                # Check step shape and calculate slope at the threshold
                is_small_start, is_small_end, threshold_slope = is_step_shape(probabilities, index)

                if is_small_start and is_small_end:
                    if threshold_slope > max_slope_at_threshold:
                        max_slope_at_threshold = threshold_slope
                        self.r_list.append(r)
                        self.b_list.append(b)
                        self.threshold_prob_list.append(probabilities[index])
                        last_r, last_b = r, b  # Track the last optimized (r, b)

                # Plot the first curve (r=1, b=1)
                if r == 1 and b == 1:
                    fig.add_trace(go.Scatter(
                        x=np.linspace(0, 1, len(probabilities)),
                        y=probabilities,
                        mode='lines',
                        name=f'First (r={r}, b={b})',
                        line=dict(width=2, dash='dash', color='green'),
                        legendrank=1
                    ))

                # Plot the last curve
                if r == max_r and b == max_b:
                    fig.add_trace(go.Scatter(
                        x=np.linspace(0, 1, len(probabilities)),
                        y=probabilities,
                        mode='lines',
                        name=f'Last (r={r}, b={b})',
                        line=dict(width=2, dash='dash', color='yellow'),
                        legendrank=2
                    ))

        # Plot all (r, b) except the last one (blue, thin)
        for opt_r, opt_b in zip(self.r_list[:-1], self.b_list[:-1]):
            probabilities = 1 - np.power((1 - np.power(self.jaccard_values, opt_r)), opt_b)
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, len(probabilities)),
                y=probabilities,
                mode='lines',
                name=f'(r={opt_r}, b={opt_b})',
                line=dict(width=1, dash='solid', color='blue'),
                legendrank=4
            ))

        if last_r is not None and last_b is not None:
            probabilities = 1 - np.power((1 - np.power(self.jaccard_values, last_r)), last_b)
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, len(probabilities)),
                y=probabilities,
                mode='lines',
                name=f'Optimized (r={last_r}, b={last_b})',
                line=dict(width=4, dash='solid', color='red'),
                legendrank=3
            ))

        # Add a vertical line at the threshold point
        fig.add_shape(
            dict(
                type='line',
                x0=threshold,
                y0=0,
                x1=threshold,
                y1=1,
                line=dict(color='black', width=2, dash='dash')
            )
        )

        # Update layout
        fig.update_layout(
            xaxis_title='Jaccard Similarity',
            yaxis_title='Probability of becoming a candidate',
            title=f'S-curve for First, Last, Chosen (optimized) and all the other (r, b) pairs',
            legend=dict(x=1, y=0.5),
            width=1600,
            height=1000
        )
        print(
            f"\nFor S-curve analysis, r: {last_r} and b: {last_b} were chosen as the last optimized (highest slope).")
        fig.show()

        # Update the chosen r, b, and threshold_prob in the class
        if last_r is not None and last_b is not None:
            probabilities = 1 - np.power((1 - np.power(self.jaccard_values, last_r)), last_b)
            threshold_index = int(threshold * len(probabilities))
            self.set_r_b_values(last_r, last_b)
        else:
            print("No suitable r and b were found that meet the criteria.")


def find_threshold_intersection(probabilities, threshold):
    # Find the index where the probabilities first cross or reach the threshold
    intersection_index = np.argmax(probabilities >= threshold)

    # If the threshold is not reached, return None
    if probabilities[intersection_index] < threshold:
        return None

    return intersection_index
