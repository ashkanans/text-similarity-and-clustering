class Shingling:
    def __init__(self, tokens, k):
        """
        Initializes the Shingling class for a tokenized document.

        Args:
            tokens (list of str): A tokenized document (list of words).
            k (int): The number of consecutive tokens in each shingle.
        """
        self.tokens = tokens
        self.k = k
        self.shingles = self.generate_shingles()

    def generate_shingles(self):
        """
        Generates a set of shingles (k-grams) from the tokenized document.

        Returns:
            set: A set of shingles, where each shingle is a tuple of k tokens.
        """
        shingles = set()
        for i in range(len(self.tokens) - self.k + 1):
            shingle = tuple(self.tokens[i:i + self.k])  # Create a tuple of k consecutive tokens
            shingles.add(shingle)
        return shingles

    def sort_shingles(self):
        """
        Sorts the shingles and returns the sorted list.

        Returns:
            list: A sorted list of shingles.
        """
        return sorted(self.shingles)
