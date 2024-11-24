import numpy as np
import hashlib
from tqdm import tqdm  # Import the tqdm library for progress bar


def hashFamily(i):
    """
    Implements a family of hash functions, parametrized by `i`.
    Uses SHA-1 for hashing and includes a salt derived from `i`.

    Args:
        i (int): Parameter to define the member of the hash function family.

    Returns:
        function: A hash function specific to the parameter `i`.
    """
    resultSize = 8  # Number of bytes in the hash value
    maxLen = 20  # Maximum length of the salt
    salt = str(i).zfill(maxLen)[-maxLen:]  # Generate salt based on `i`

    def hashMember(x):
        # Hashes the input using SHA-1 with the salt and returns the last `resultSize` bytes as an integer.
        return int.from_bytes(hashlib.sha1((str(x) + salt).encode('utf-8')).digest()[-resultSize:], 'big')

    return hashMember


class MinwiseHashSignature:
    """
    Implements Min-Hashing to generate compact signatures for sets,
    approximating Jaccard similarity between the sets.

    Attributes:
        num_hashes (int): Number of hash functions to use.
        signature_matrix (numpy.ndarray): The signature matrix storing the Min-Hash values.
        hash_functions (list): List of hash functions from the `hashFamily`.
    """

    def __init__(self, num_hashes, num_elements):
        self.num_hashes = num_hashes
        self.num_elements = num_elements
        self.signature_matrix = np.full((num_hashes, num_elements), np.inf)
        self.hash_functions = [hashFamily(i) for i in range(num_hashes)]

    def _hash_set(self, set_elements):
        """
        Efficiently hashes all elements of a set with all hash functions.

        Args:
            set_elements (set): The set of elements (e.g., shingles) for the document.

        Returns:
            np.ndarray: A (num_hashes, len(set_elements)) matrix of hashed values.
        """
        if not set_elements:
            # Return an empty matrix if set_elements is empty
            return np.zeros((self.num_hashes, 0), dtype=np.uint64)

        hashed_matrix = np.zeros((self.num_hashes, len(set_elements)), dtype=np.uint64)

        for i, element in enumerate(set_elements):
            for j, hash_function in enumerate(self.hash_functions):
                hashed_matrix[j, i] = hash_function(element)

        return hashed_matrix

    def _update_signature_matrix(self, set_index, hashed_matrix):
        """
        Updates the signature matrix with minimum hash values for a set.

        Args:
            set_index (int): Index of the current set (e.g., document).
            hashed_matrix (np.ndarray): A (num_hashes, len(set_elements)) matrix of hashed values.
        """
        if hashed_matrix.size == 0:
            # Skip update if hashed_matrix is empty
            return

        # Take the minimum value across columns for each hash function
        self.signature_matrix[:, set_index] = np.minimum(
            self.signature_matrix[:, set_index],
            np.min(hashed_matrix, axis=1)
        )

    def generate_signatures(self, sets):
        """
        Generates the Min-Hash signatures for all sets.

        Args:
            sets (list of sets): List of sets (e.g., shingle sets for documents).

        Returns:
            np.ndarray: The signature matrix.
        """
        print("Generating signatures (optimized min-hashing) from shingles...")

        # Use tqdm to show progress bar
        for set_index, set_elements in tqdm(enumerate(sets), total=len(sets), desc="Processing Sets"):
            hashed_matrix = self._hash_set(set_elements)
            self._update_signature_matrix(set_index, hashed_matrix)

        return self.signature_matrix

    def hash_values_to_buckets(self, hash_values):
        """
        Maps hash values into buckets for use in Locality-Sensitive Hashing (LSH).

        Args:
            hash_values (np.ndarray): Array of hash values for a single signature.

        Returns:
            list: List of tuples representing buckets.
        """
        buckets = []
        for i in range(0, len(hash_values), self.num_hashes):
            bucket = tuple(hash_values[i:i + self.num_hashes])
            buckets.append(bucket)
        return buckets
