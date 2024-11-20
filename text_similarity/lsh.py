import time

import numpy as np
import plotly.graph_objects as go


class LSH:
    def __init__(self, minwise_hash_signature, shingle_comparison, num_buckets):
        self.threshold_prob_list = []
        self.b_list = []
        self.r_list = []
        self.jaccard_values = None
        self.elapsed_time_lsh = None
        self.candidates = None
        self.threshold_prob = 0
        self.r = 0
        self.b = 0
        self.minwise_hash_signature = minwise_hash_signature
        self.shingle_comparison = shingle_comparison
        self.num_buckets = num_buckets
        self.hash_tables = [{} for _ in range(minwise_hash_signature.num_hashes)]

    def index_signatures(self, signatures):
        for i, signature in enumerate(signatures.T):
            hash_values = self.minwise_hash_signature.hash_element(signature)
            buckets = self.minwise_hash_signature.hash_values_to_buckets(hash_values)

            for j, bucket in enumerate(buckets):
                if bucket not in self.hash_tables[j]:
                    self.hash_tables[j][bucket] = []
                self.hash_tables[j][bucket].append(i)

    def query_signatures(self, query_signature):
        hash_values = self.minwise_hash_signature.hash_element(query_signature)
        buckets = self.minwise_hash_signature.hash_values_to_buckets(hash_values)

        candidate_sets = set()
        for i, bucket in enumerate(buckets):
            if bucket in self.hash_tables[i]:
                candidate_sets.update(self.hash_tables[i][bucket])

        return candidate_sets

    def find_near_duplicates(self, shingles):
        start_time = time.time()
        signatures = self.minwise_hash_signature.generate_signatures(shingles)
        self.index_signatures(signatures)

        # Perform LSH query for a specific document (e.g., the first document)
        query_signature = signatures[:, 0]
        candidates = self.query_signatures(query_signature)
        end_time = time.time()
        print("LSH Near Duplicates:", candidates)
        elapsed_time_lsh = end_time - start_time
        self.candidates = candidates
        self.elapsed_time_lsh = elapsed_time_lsh

    def set_r_b_values(self, r, b, threshold_prob):
        self.r = r
        self.b = b
        self.threshold_prob = threshold_prob

    def s_curve_plot_and_analysis(self, jaccard_values, max_r, max_b, threshold=0.8):
        """
        Plot S-curves showing the first, last, and all optimized (r, b) combinations.
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
                        line=dict(width=2, dash='dash', color='green')
                    ))

                # Plot the last curve
                if r == max_r and b == max_b:
                    fig.add_trace(go.Scatter(
                        x=np.linspace(0, 1, len(probabilities)),
                        y=probabilities,
                        mode='lines',
                        name=f'Last (r={r}, b={b})',
                        line=dict(width=2, dash='dash', color='yellow')
                    ))

        # Plot all optimized (r, b) except the last one (blue, thin)
        for opt_r, opt_b in zip(self.r_list[:-1], self.b_list[:-1]):
            probabilities = 1 - np.power((1 - np.power(self.jaccard_values, opt_r)), opt_b)
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, len(probabilities)),
                y=probabilities,
                mode='lines',
                name=f'Optimized (r={opt_r}, b={opt_b})',
                line=dict(width=1, dash='solid', color='blue')
            ))

        # Highlight the last optimized (r, b) (red, thick)
        if last_r is not None and last_b is not None:
            probabilities = 1 - np.power((1 - np.power(self.jaccard_values, last_r)), last_b)
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, len(probabilities)),
                y=probabilities,
                mode='lines',
                name=f'Last Optimized (r={last_r}, b={last_b})',
                line=dict(width=4, dash='solid', color='red')
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
            title=f'S-curve for First, All Optimized, and Last Optimized (r, b)',
            legend=dict(x=1, y=0.5),
            width=1200,
            height=800
        )
        print(
            f"\nFor S-curve analysis, r: {last_r} and b: {last_b} were chosen as the last optimized (highest slope).")
        fig.show()

        # Update the chosen r, b, and threshold_prob in the class
        if last_r is not None and last_b is not None:
            probabilities = 1 - np.power((1 - np.power(self.jaccard_values, last_r)), last_b)
            threshold_index = int(threshold * len(probabilities))
            self.set_r_b_values(last_r, last_b, probabilities[threshold_index])
        else:
            print("No suitable r and b were found that meet the criteria.")

    def choose_r_b_values(self, threshold_prob):
        # Choose r and b values based on the given threshold probability

        for r_candidate in range(1, self.minwise_hash_signature.num_hashes + 1):
            for b_candidate in range(1, self.minwise_hash_signature.num_hashes // r_candidate + 1):
                # Calculate the expected threshold probability for the given r and b
                expected_prob = 1 - (1 - threshold_prob ** r_candidate) ** b_candidate

                # Check if the expected probability is close enough to the given threshold probability
                if abs(expected_prob - threshold_prob) < 0.01:
                    self.r = r_candidate
                    self.b = b_candidate
                    self.threshold_prob = expected_prob
                    return


def find_threshold_intersection(probabilities, threshold):
    # Find the index where the probabilities first cross or reach the threshold
    intersection_index = np.argmax(probabilities >= threshold)

    # If the threshold is not reached, return None
    if probabilities[intersection_index] < threshold:
        return None

    return intersection_index
