class Shingling:
    def __init__(self, documents, k, is_tokenized):
        """
        Initializes the Shingling class for a tokenized document.

        Args:
            is_tokenized: the document is tokenized
            documents (list of str): A tokenized document (list of words).
            k (int): The number of consecutive tokens in each shingle.
        """
        self.is_tokenized = is_tokenized
        self.documents = documents
        self.k = k
        self.shingles = self.generate_shingles()

    def generate_shingles(self):
        """
        Generates shingles from the documents.

        If `is_tokenized` is False, creates shingles directly from raw descriptions (character-based).
        If `is_tokenized` is True, creates shingles from tokenized descriptions.

        Returns:
            list: A list where each element is a set of shingles for the corresponding document.
        """
        shingles_list = []

        for doc in self.documents:
            if self.is_tokenized:
                # Generate token-based shingles from tokenized documents
                shingles = {
                    " ".join(doc[i:i + self.k]) for i in range(len(doc) - self.k + 1)
                }
            else:
                # Generate character-based shingles from raw descriptions
                shingles = {
                    doc[i:i + self.k] for i in range(len(doc) - self.k + 1)
                }

            shingles_list.append(shingles)

        return shingles_list
