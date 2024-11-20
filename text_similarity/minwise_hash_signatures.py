
import numpy as np
import hashlib


class MinwiseHashSignature:
    def __init__(self, num_hashes, num_elements, signature_length):
        self.num_hashes = num_hashes
        self.num_elements = num_elements
        self.signature_length = signature_length
        self.signature_matrix = np.full((num_hashes, num_elements), fill_value=float('inf'))
        self.hash_functions = [self.generate_hash_function() for _ in range(num_hashes)]

    def generate_hash_function(self):
        def hash_function(x):
            return int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16)

        return hash_function

    def hash_element(self, element):
        hashed_values = [hash_function(element) for hash_function in self.hash_functions]
        return np.array(hashed_values)

    def update_signature_matrix(self, set_index, set_elements):
        for i in range(self.num_hashes):
            for element in set_elements:
                hash_value = self.hash_element(element)
                self.signature_matrix[i, set_index] = min(self.signature_matrix[i, set_index], hash_value[i])

    def generate_signatures(self, sets):
        for i, set_elements in enumerate(sets):
            self.update_signature_matrix(i, set_elements)

        return self.signature_matrix

    def hash_values_to_buckets(self, hash_values):
        buckets = []
        for i in range(0, len(hash_values), self.num_hashes):
            bucket = tuple(hash_values[i:i + self.num_hashes])
            buckets.append(bucket)
        return buckets
