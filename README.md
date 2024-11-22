
# Text Similarity and Clustering

### Text Similarity

The objective of this task is to find near-duplicate products in a collection of Amazon product descriptions. To achieve this, we are tasked with implementing a nearest-neighbor search using text documents. Specifically, we will use **shingling**, **minwise hashing**, and **locality-sensitive hashing (LSH)** to identify products that are similar based on their textual content.

The problem involves the following steps:

1. **Shingling**: Create shingles (substrings) of length 10 from the product descriptions.
2. **Minwise Hashing**: Use minwise hashing to create signatures for the shingles.
3. **Locality-Sensitive Hashing (LSH)**: Implement LSH to efficiently find pairs of documents with a high similarity, specifically with a Jaccard coefficient of at least 80%.
4. **Comparison**: Compare the performance of LSH with a brute-force method that calculates the Jaccard similarity between all pairs of documents.

The task evaluates the ability to efficiently find near-duplicates using advanced hashing techniques and compares the results in terms of both accuracy and computational efficiency.

### Analyzing the Amazon Products Description


### Preprocessing Descriptions for Near-Duplicate Search

When scraping Amazon for PC or laptop listings, the scraped data is saved as a file named `computer_results_default.tsv` in the `data/raw` directory. To address the near-duplicate search problem, we perform the search in two ways:

1. **On raw descriptions**: Comparing the unprocessed descriptions directly.
2. **On preprocessed descriptions**: Applying preprocessing to standardize and clean the descriptions before comparison.

Here’s a concise explanation of how preprocessing is performed using the `TextPreprocessor` class and the `preprocess_descriptions` function:

---

### **Steps in Preprocessing**

#### **1. Multi-Word Term Preservation**
   - Phrases like "windows 10 pro" or "dual band wifi" are preserved as single tokens by replacing spaces with underscores (e.g., `"windows_10_pro"`).

#### **2. Tokenization**
   - The text is split into individual tokens (words and punctuation) using the NLTK tokenizer.

#### **3. Punctuation and Symbol Removal**
   - Non-alphanumeric characters are removed. However:
     - Mixed alphanumeric terms (e.g., `"16GB"`) are retained.
     - Pure numbers and specific units (e.g., `"gb"`, `"mhz"`) are filtered out.

#### **4. Handling Joined Terms**
   - Certain word pairs (e.g., `"m2"` and `"ssd"`) are combined into meaningful phrases like `"m.2 ssd"`.

#### **5. Stopword Removal**
   - Common English stopwords (e.g., "the", "is") and some Italian stopwords (e.g., "di", "per") are removed, unless explicitly configured to remain.

#### **6. Token Processing (Stemming and Lemmatization)**
   - Tokens in a list of preserved words (e.g., "laptop", "windows") are lemmatized (converted to their base forms).
   - Other tokens are stemmed (reduced to their root forms) for standardization.

#### **7. Multi-Word Term Restoration**
   - Multi-word terms previously preserved with underscores are restored to their original format (e.g., `"windows_10_pro"` becomes `"windows 10 pro"`).

---

### **Final Output**

The preprocessing produces a list of cleaned, normalized tokens tailored for further analysis, such as text classification or search optimization.

For instance, given an input like:
```text
"Windows 10 Pro laptop with 16GB DDR4 RAM and 512GB SSD."
```
The output might look like:
```python
['windows_10_pro', 'laptop', '16gb', 'ddr4', 'ram', '512gb', 'ssd']
```

---

### **Impact on Near-Duplicate Search**

Preprocessing standardizes the descriptions, potentially may improve the accuracy of near-duplicate detection. We will investigate this in for each of the near-duplicate search we perform.


### Finding Near-Duplicates (using LSH)

#### S-curves showing the first, last, the chosen and all (r, b) combinations.

### Finding Near-Duplicates (by comparing them with each other)


