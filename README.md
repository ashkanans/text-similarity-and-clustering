
# Text Similarity and Clustering

# Text Similarity

This project aims to identify near-duplicate products within a dataset of Amazon product descriptions by implementing a
**nearest-neighbor search** using text-based methods. The task focuses on leveraging techniques such as **shingling**, *
*minwise hashing**, and **locality-sensitive hashing (LSH)** to detect similarities in textual content efficiently.

### Approach Overview

The solution is structured into the following key steps:

1. **Shingling**: Generate shingles (substrings) of fixed length (10) from the product descriptions to represent the
   textual content in a granular way.
2. **Minwise Hashing**: Create compact signatures for the shingles using minwise hashing, which enables efficient
   similarity comparisons.
3. **Locality-Sensitive Hashing (LSH)**: Implement LSH to identify document pairs with a high similarity, focusing on
   those with a **Jaccard coefficient of at least 80%**.
4. **Performance Comparison**:
    - Benchmark LSH against a brute-force approach that computes Jaccard similarity for all document pairs.
    - Compare the implemented LSH with the DataSketch LSH library to evaluate its accuracy and efficiency.

### LSH Pipeline

Here is a visual representation of the LSH pipeline:

![LSH Pipeline](files/text_similarity/lsh%20pipeline.png)

### Analyzing Amazon Product Descriptions

Amazon product descriptions present several challenges that complicate analysis and similarity detection. These include
duplication of information, inconsistent formatting, incomplete data, localization complexities, and repetitive
patterns. However, even with a quick glance, it's possible to identify near-duplicate products, revealing subtle
variations such as changes in RAM size or keyboard color.

For example:

- **Gold Version:** *PC Portatile Computer Portatile Win11 Notebook 14 Pollici 6GB 256GB SSD Expansion 1TB, Laptop
  Celeron N4020 fino a 2.8GHz, 5000mAh 5G WiFi BT4.2 Supporto Mouse Wireless & Protezione Tastiera-Oro*
- **Red Version:** *PC Portatile Computer Portatile Win11 Notebook 14 Pollici 6GB 256GB SSD Expansion 1TB, Laptop
  Celeron N4020 fino a 2.8GHz, 5000mAh 5G WiFi BT4.2 Supporto Mouse Wireless & Protezione Tastiera-Rosso*


### Preprocessing Descriptions for Near-Duplicate Search

When scraping Amazon for PC or laptop listings, the scraped data is saved as a file named `computer_results_default.tsv` in the `data/raw` directory. To address the near-duplicate search problem, we perform the search in two ways:

1. **On raw descriptions**: Comparing the unprocessed descriptions directly.
2. **On preprocessed descriptions**: Applying preprocessing to standardize and clean the descriptions before comparison.

Here’s a concise explanation of how preprocessing is performed using the `TextPreprocessor` class and the `preprocess_descriptions` function:

---

### **Steps in Preprocessing**

#### **1. Multi-Word Term Preservation**

- Phrases like `"windows 10 pro"` or `"dual band wifi"` are preserved as single tokens by replacing spaces with
  underscores (e.g., `"windows_10_pro"`).

#### **2. Tokenization**
   - The text is split into individual tokens (words and punctuation) using the NLTK tokenizer.

#### **3. Punctuation and Symbol Removal**
   - Non-alphanumeric characters are removed. However:
     - Mixed alphanumeric terms (e.g., `"16GB"`) are retained.
     - Numeric values with decimals (e.g., `4.6 GHz`, `15.6 inch`) and certain units (e.g., `"gb"`, `"mhz"`) are
       preserved.

#### **4. Handling Joined Terms**
   - Certain word pairs (e.g., `"m2"` and `"ssd"`) are combined into meaningful phrases like `"m.2 ssd"`.

#### **5. Stopword Removal**
   - Common English stopwords (e.g., "the", "is") and some Italian stopwords (e.g., "di", "per") are removed, unless explicitly configured to remain.

#### **6. Token Processing (Stemming and Lemmatization)**

- Tokens in a list of preserved words (e.g., `"laptop"`, `"windows"`) are lemmatized (converted to their base forms).
   - Other tokens are stemmed (reduced to their root forms) for standardization.

#### **7. Multi-Word Term Restoration**
   - Multi-word terms previously preserved with underscores are restored to their original format (e.g., `"windows_10_pro"` becomes `"windows 10 pro"`).

---

### **Example Preprocessing**

For different input examples, the preprocessing steps produce the following outputs:

#### **Example 1: Lenovo Notebook Description**

**Raw Description:**

```plaintext
Lenovo Notebook Nuovo • 24gb di Ram • CPU Intel Core i5 • Monitor 15.6 Full HD • SSD 256 nvme + SSD 128 GB SATA • Sistema operativo Win 11 e Libre Office • Mouse + Adattatore usb Type C
```

**Output (Tokenized=False):**

```plaintext
'lenovo notebook nuovo 24gb ram cpu intel core i5 monitor 15.6 full hd ssd 256 nvme ssd 128 gb sata sistema operativo win 11 libre offic mouse adattator usb type c'
```

**Output (Tokenized=True):**

```python
['lenovo', 'notebook', 'nuovo', '24gb', 'ram', 'cpu', 'intel core i5', 'monitor', '15.6', 'full hd', 'ssd', '256', 'nvme', 'ssd', '128', 'gb', 'sata', 'sistema', 'operativo', 'win 11', 'libre offic', 'mouse', 'adattator', 'usb', 'type', 'c']
```

---

#### **Example 2: Samsung Galaxy Book Description**

**Raw Description:**

```plaintext
Samsung Galaxy Book4 Pro 360, Intel® Core™ Ultra 7 Processor, 16GB RAM, 512GB, Laptop 16 Dynamic AMOLED 2X touch, S Pen, Windows 11 Home, Moonstone Gray [Versione Italiana]
```

**Output (Tokenized=False):**

```plaintext
'samsung galaxi book4 pro 360 intel core ultra 7 processor 16gb ram 512gb laptop 16 dynam amoled 2x touch pen windows 11 hom moonston gray version italiana'
```

**Output (Tokenized=True):**
```python
['samsung', 'galaxi', 'book4', 'pro', '360', 'intel', 'core', 'ultra', '7', 'processor', '16gb ram', '512gb', 'laptop', '16', 'dynam', 'amoled', '2x', 'touch', 'pen', 'windows 11 hom', 'moonston', 'gray', 'version', 'italiana']
```

---

#### **Example 3: ASUS Expertbook Description**

**Raw Description:**

```plaintext
ASUS Expertbook i5-1335u 4.6 GHz 15.6 pollici FHD Ram 16 Gb Ddr4 Ssd Nvme 512 Gb HDMI USB 3.0 WiFi Bluetooth Webcam Office Pro 2021 Windows 11 Pro
```

**Output (Tokenized=False):**

```plaintext
'asus expertbook i5-1335u 4.6 ghz 15.6 inch full hd ram 16 gb ddr4 ssd nvme 512 gb hdmi usb 3.0 wifi bluetooth webcam office pro 2021 windows 11 pro'
```

**Output (Tokenized=True):**

```python
['asus', 'expertbook', 'i5-1335u', '4.6', 'ghz', '15.6 inch', 'full', 'hd', 'ram', '16', 'gb', 'ddr4', 'ssd', 'nvme', '512', 'gb', 'hdmi', 'usb 3.0', 'wifi', 'bluetooth', 'webcam', 'office', 'pro', '2021', 'windows 11 pro']
```

---

### **Key Takeaways**

- Preprocessing helps standardize descriptions for better near-duplicate detection.
- The `TextPreprocessor` class ensures consistent tokenization, cleaning, and normalization.
- Numeric values, abbreviations, and multi-word terms are carefully preserved to maintain essential context.

---

#### **Impact on Near-Duplicate Search**

Preprocessing standardizes the descriptions, potentially may improve the accuracy of near-duplicate detection. We will investigate this in for each of the near-duplicate search we perform.

### Shingling Process

The shingling process involves transforming document descriptions into sets of fixed-length substrings, or "shingles," to facilitate the comparison of textual similarity. This method captures the structural patterns within documents, enabling accurate similarity calculations even when text order or structure varies slightly.

#### Implementation Details

1. **Class Overview**:
   - The `Shingling` class is designed to generate shingles from document descriptions, either as sequences of characters or as sequences of words (tokens).
   - It takes three key inputs:
     - A list of document descriptions (`documents`).
     - A shingle length (`k`), defining the size of each shingle.
     - A flag (`is_tokenized`) to determine whether input descriptions are tokenized (word-based) or raw text (character-based).

2. **Shingle Creation**:
   - **Character-Based Shingles**: 
     - For raw text, \( k \)-length shingles are created by sliding a window of size \( k \) across the text.
     - Example:
       ```
       Document: "laptop computer"
       k = 5 → Shingles: {"lapto", "aptop", "ptop ", "top c", "op co", "p com", ...}
       ```
   - **Token-Based Shingles**:
     - For tokenized descriptions, shingles are generated as sequences of \( k \)-consecutive words joined by spaces.
     - Example:
       ```
       Tokens: ["laptop", "computer", "portable"]
       k = 2 → Shingles: {"laptop computer", "computer portable"}
       ```

3. **Output**:
   - Each document is represented as a set of unique shingles.
   - The output is a list where each entry corresponds to the set of shingles for a single document.

4. **Optional Sorting**:
   - The `sort_shingles` method provides an option to sort the shingles for debugging or visualization purposes.

#### Advantages of This Approach
- **Flexibility**: Supports both character-level and token-level shingles, allowing adaptation to the type of input data and the granularity of similarity detection.
- **Compact Representation**: By using sets, duplicate shingles within the same document are avoided, optimizing storage and processing.
- **Order Sensitivity**: Shingles inherently capture the order of characters or tokens, making the method robust to textual rearrangements.

This shingling implementation serves as the foundational step for the Min-Hashing and Locality-Sensitive Hashing (LSH) processes, ensuring efficient and scalable similarity detection.

### Min-Hashing Process

Min-Hashing is a critical stage in the pipeline for detecting near-duplicate documents. It compresses large sets of shingles into compact numerical signatures while preserving the approximate Jaccard similarity between the original sets. These signatures enable efficient comparison of large datasets.

#### Implementation Details

1. **Class Overview**:
   - The `MinwiseHashSignature` class implements the Min-Hashing process.
   - It uses a family of hash functions, as described in the homework, to compute compact and similarity-preserving signatures for each document.

2. **Key Components**:
   - **Number of Hash Functions (`num_hashes`)**:
     - Specifies how many distinct hash functions are used to create the signature for each document.
   - **Signature Matrix**:
     - A matrix of shape `(num_hashes, num_documents)` where:
       - Each column represents the Min-Hash signature for a document.
       - Each row corresponds to the minimum hash value obtained for a particular hash function across the shingles of a document.

3. **Hash Function Family**:
   - The homework-provided `hashFamily` function is used to generate distinct hash functions:
     - Each hash function is parameterized by an integer \( i \), ensuring uniqueness.
     - SHA-1 hashing is used with an \( i \)-derived salt to produce stable and independent hash values for shingles.

4. **Signature Computation**:
   - For each document:
     - All shingles are hashed using the family of hash functions.
     - The minimum hash value for each function is stored in the signature matrix.
   - This ensures that the resulting signature matrix compactly represents the shingle sets while preserving their similarity.

5. **Output**:
   - The final signature matrix contains the Min-Hash signatures for all documents.
   - These signatures approximate the Jaccard similarity between documents and are ready for the Locality-Sensitive Hashing (LSH) stage.

#### Advantages of Min-Hashing
- **Dimensionality Reduction**:
  - Transforms large, sparse sets of shingles into fixed-size signature vectors.
  - Significantly reduces memory and computational requirements for similarity comparisons.
- **Similarity Preservation**:
  - Ensures that the probability of two documents having the same Min-Hash value equals their Jaccard similarity.
- **Scalability**:
  - Enables efficient processing of large datasets by reducing the need for pairwise comparisons.

#### Example
Given the following documents:
- Document 1: "laptop computer portable"
- Document 2: "portable laptop gaming"

Using `k = 2`  (shingle length), the shingles and signatures are generated as follows:
- Shingles (Document 1): {"laptop computer", "computer portable"}
- Shingles (Document 2): {"portable laptop", "laptop gaming"}

With 100 hash functions, the Min-Hashing stage generates compact signatures (e.g., `[12, 34, 56, ...]` for each document).

The Min-Hashing stage ensures that the signatures preserve the approximate Jaccard similarity between the original shingle sets, preparing the data for the Locality-Sensitive Hashing (LSH) stage.

### Finding Near-Duplicates (using LSH)

Locality-Sensitive Hashing (LSH) is a crucial stage in the pipeline for efficiently finding near-duplicate documents. It leverages the compact Min-Hash signatures from the previous stage to identify candidate pairs with high similarity, avoiding costly pairwise comparisons.

#### Implementation Details

The `LSH` class implements this process by using hash tables and banding techniques to filter and group similar documents. Below are the steps taken by the class:

---

#### 1. **Initialization**
   - The `LSH` class is initialized with:
     - A `minwise_hash_signature` object to provide the Min-Hash signatures.
     - A `shingle_comparison` object for validating similarity based on Jaccard coefficients.
     - `num_buckets`: The number of buckets for hashing signature bands.
   - Internally, it creates:
     - A list of hash tables (`hash_tables`) to store the buckets for each band.

---

#### 2. **Indexing Signatures**
   - **Purpose**: Maps document signatures into buckets for efficient similarity grouping.
   - **Process**:
     - Each signature is split into bands (groups of rows from the signature matrix).
     - Each band is hashed into a bucket, ensuring similar bands hash to the same bucket.
     - Documents with matching bucket values in any band are considered potential candidate pairs.

---

#### 3. **Querying Signatures**
   - **Purpose**: Retrieve candidate pairs for a given query document.
   - **Process**:
     - The query signature is split into bands and hashed into buckets.
     - Candidate documents are identified by finding matching buckets in the hash tables.
   - **Efficiency**: Only a subset of all documents is considered, greatly reducing computational costs.

---

#### 4. **Finding Near-Duplicates**
   - **Purpose**: Identifies near-duplicate documents using LSH.
   - **Process**:
     1. Generate Min-Hash signatures for all documents.
     2. Index these signatures into hash tables.
     3. Query hash tables for potential candidates (documents with matching bucket values).
     4. Validate the similarity of candidate pairs using exact Jaccard calculations if needed.
   - **Output**:
     - A list of candidate pairs that are likely near-duplicates.
     - The time taken to perform the operation (`elapsed_time_lsh`).

---

#### 5. **S-Curve Analysis for Parameter Optimization**
   - **Purpose**: Tune the LSH parameters (\( r \): rows per band, \( b \): number of bands) for the best trade-off between false positives and false negatives.
   - **Process**:
     1. Generate S-curves representing the probability of candidate selection as a function of similarity for different \( r, b \) combinations.
     2. Identify \( r, b \) values that maximize the slope of the S-curve at the desired similarity threshold (e.g., \( s = 0.8 \)).
   - **Visualization**:
     - Plots showing S-curves for the first, last, and optimized \( r, b \) combinations.
     - Vertical lines indicate the similarity threshold.

---

#### 6. **Setting \( r, b \) Values**
   - After identifying the optimal \( r, b \) values, these are used in subsequent LSH operations.
   - Ensures that only documents with high similarity (e.g., \( \geq 0.8 \)) are retrieved as candidates.

---

#### Advantages of LSH
1. **Scalability**:
   - Drastically reduces the number of comparisons required to find near-duplicates.
2. **Parameter Optimization**:
   - The S-curve analysis fine-tunes \( r, b \) values for precise control over false positives and false negatives.
3. **Adaptability**:
   - Supports customization through the number of buckets, \( r, b \), and similarity thresholds.
4. **Efficiency**:
   - Avoids exhaustive pairwise comparisons, enabling rapid similarity detection in large datasets.

---

#### Example Workflow with LSH
1. **Input**: Min-Hash signatures for a collection of Amazon product descriptions.
2. **Process**:
   - Split signatures into bands and hash into buckets.
   - Identify candidate pairs by finding documents that share buckets in at least one band.
   - Validate candidate pairs using Jaccard similarity.
3. **Output**:
   - Candidate pairs of near-duplicate products.
   - Execution time and performance metrics.
4. **Visualization**:
   - An S-curve plot demonstrating the probability of detecting similar pairs at varying similarity thresholds.

---

The LSH implementation in this pipeline ensures efficient, scalable, and accurate identification of near-duplicate documents, making it ideal for large datasets such as Amazon product descriptions.

#### S-curves showing the first, last, the chosen and all (r, b) combinations.

### Finding Near-Duplicates (by comparing them with each other)
