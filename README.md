
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

### S-Curve Analysis for Parameter Optimization

---

#### **Purpose**

The S-curve analysis is conducted to optimize the parameters of the LSH algorithm (\( r \): rows per band, \( b \):
number of bands) for achieving a balance between false positives and false negatives. This ensures effective similarity
detection while maintaining computational efficiency.

---

#### **Process**

1. **Generating S-Curves**:
    - Multiple S-curves are plotted for various \( r, b \) combinations, illustrating the probability of candidate
      selection as a function of Jaccard similarity.
    - Steeper slopes at the similarity threshold (\( s = 0.8 \)) indicate better performance in distinguishing similar
      and dissimilar pairs.

2. **Parameter Tuning**:
    - The optimal \( r, b \) combination is identified as the one that maximizes the slope of the S-curve at \( s =
      0.8 \).
    - Two cases were explored:
        - **Moderate Parameters (\( r = 16, b = 20 \))**: Faster execution with acceptable accuracy.
        - **High Parameters (\( r = 20, b = 50 \))**: Improved accuracy but slower due to computational overhead.

---

#### **Results**

1. **Moderate Parameters: \( r = 16, b = 20 \)**
    - **Observation**: The S-curve has a reasonably steep slope at \( s = 0.8 \), indicating a good balance between
      efficiency and accuracy.
    - **Visualization**: The plot below shows the S-curve for this setting, highlighting the optimized (red) curve.

   ![S-Curve Analysis - (r=16, b=20)](files/text_similarity/S-curve%20analysis%20-%20(r=16,b=20).png)

2. **High Parameters: \( r = 20, b = 50 \)**
    - **Observation**: The S-curve for \( r = 20, b = 50 \) has a much steeper slope at \( s = 0.8 \), reflecting higher
      precision in candidate selection. However, this setting results in slower execution due to increased computational
      demands.
    - **Visualization**: The plot below demonstrates the effect of increasing \( r, b \).

   ![S-Curve Analysis - (r=20, b=50)](files/text_similarity/S-curve%20analysis%20-%20(r=20,b=50).png)

3. **Comparison of Moderate and High Parameters**
    - **Observation**: The difference between the two settings is shown in the figure below. The red curve (\( r=20,
      b=50 \)) is steeper at the threshold, offering better precision, while the blue curve (\( r=16, b=20 \)) provides
      a more computationally efficient solution with a slightly lower slope.

   ![Difference Between (r=16, b=20) and (r=20, b=50)](files/text_similarity/the%20difference%20between%20different%20(r,b)%20pairs%20-%20(r=16,b=20)%20vs%20(r=20,%20b=50).png)

---

#### **Summary**

- **Moderate Parameters (\( r = 16, b = 20 \))**:
    - Faster execution, suitable for large datasets where computational cost is a concern.
    - A good trade-off between accuracy and efficiency.

- **High Parameters (\( r = 20, b = 50 \))**:
    - Higher precision, capturing more near-duplicates, but slower execution.
    - Useful in scenarios where accuracy is critical, and computation time is less of a concern.

By leveraging S-curve analysis, the appropriate parameter configuration can be chosen based on the specific requirements
of the application.

#### Advantages of LSH

Until now, we have covered also all the details about out LSH pipeline. Just to summrize the key advantages of our LSH
implementation:
1. **Scalability**:
   - Drastically reduces the number of comparisons required to find near-duplicates.
2. **Parameter Optimization**:
   - The S-curve analysis fine-tunes \( r, b \) values for precise control over false positives and false negatives.
3. **Adaptability**:
   - Supports customization through the number of buckets, \( r, b \), and similarity thresholds.
4. **Efficiency**:
   - Avoids exhaustive pairwise comparisons, enabling rapid similarity detection in large datasets.

---

### Example Workflow with LSH

This section describes the step-by-step process of performing similarity analysis using the LSH pipeline, from the
initial parameter tuning to the final output.

---

#### **Step 1: Parameter Optimization with S-Curve Analysis**

- The pipeline begins by performing **S-curve analysis** to fine-tune the LSH parameters \( r \) (rows per band) and \(
  b \) (number of bands).
- The S-curve visualization helps identify the optimal \( r, b \) combination by maximizing the slope at a given
  similarity threshold (e.g., \( s = 0.8 \)).
- Based on the analysis, the pipeline sets the optimal \( r, b \) values to balance computational efficiency with
  similarity detection accuracy.

---

#### **Step 2: Generating Min-Hash Signatures**

- After parameter optimization, Min-Hash signatures are generated for the input shingle sets.
- The number of hash functions is calculated as \( r \times b \), ensuring the signatures align with the chosen
  parameters.
- Min-Hash signatures provide a compact representation of the input data, enabling efficient similarity computations.

---

#### **Step 3: Indexing Min-Hash Signatures**

- The generated Min-Hash signatures are split into bands, and each band is hashed into buckets using the LSH hash
  function.
- This step organizes the data into buckets where similar items are more likely to collide, drastically reducing the
  number of comparisons needed.

---

#### **Step 4: Identifying Near-Duplicates**

- The pipeline identifies **candidate pairs** by finding items that share at least one bucket in any band.
- Candidate pairs are validated by calculating their actual Jaccard similarity to ensure they meet the similarity
  threshold and are saved in the csv file

---

### Comparison of Naïve, LSH, and DataSketch Methods for Text Similarity

This section evaluates the performance of Naïve (brute-force), LSH, and DataSketch approaches for computing text
similarity. Various metrics and visualizations are used to highlight differences and similarities across the methods.

---

#### 1. Venn Diagram: Overlap in Detected Similar Pairs

The Venn diagrams show the overlap in similar pairs detected by different methods.

- **Naïve vs LSH**:
    - Intersection: 356 pairs.
    - Unique to Naïve: 80 pairs.
    - Unique to LSH: 392 pairs.

![Naïve vs LSH](files/text_similarity/Venn%20Diagram-%20Naïve%20(brute-force)%20vs%20LSH.png)

- **Naïve vs DataSketch**:
    - Intersection: 340 pairs.
    - Unique to Naïve: 96 pairs.
    - Unique to DataSketch: 374 pairs.

![Naïve vs DataSketch](files/text_similarity/Venn%20Diagram-%20Naïve%20(brute-force)%20vs%20DataSketch.png)

---

#### 2. Jaccard Similarity Box Plot

The box plot compares the distribution of Jaccard similarity scores for the three methods:

- **Naïve** has a wider range of scores, with a few outliers near 1.0.
- **LSH and DataSketch** exhibit a more concentrated range around higher similarity values, reflecting their optimized
  nature for capturing higher similarities.

![Jaccard Similarity Box Plot](files/text_similarity/Jaccard%20Similarity%20Box%20Plot.png)

---

#### 3. Precision, Recall, and F1-Score Comparison

The bar plots highlight performance metrics:

- **LSH** achieves higher recall but lower precision than DataSketch.
- **DataSketch** offers balanced metrics with slightly better precision and F1-score.

![LSH and DataSketch Metrics Comparison](files/text_similarity/Comparison%20of%20LSH%20and%20DataSketch%20Metrics.png)

---

#### 4. Cumulative Distribution of Jaccard Similarities

The cumulative distribution plot compares how similarity scores are distributed:

- **Naïve** leads in identifying pairs with scores above 0.95.
- **LSH and DataSketch** produce more consistent results over the full range of similarities, with smoother cumulative
  curves.

![Cumulative Distribution of Jaccard Similarities](files/text_similarity/Cumulative%20Distribution%20of%20Jaccard%20Similarities.png)

---

#### 5. Jaccard Similarity Heatmaps

The heatmaps visualize pairwise Jaccard similarities:

- **Naïve** shows denser clusters, reflecting all pair comparisons.
- **LSH and DataSketch** produce sparser but well-distributed similarities, focusing on near-duplicates.

- **Naïve**:
  ![Naïve Jaccard Similarity Heatmap](files/text_similarity/Naïve%20(brute-force)%20Jaccard%20Similarity%20Heatmap.png)

- **LSH**:
  ![LSH Jaccard Similarity Heatmap](files/text_similarity/LSH%20Jaccard%20Similarity%20Heatmap.png)

- **DataSketch**:
  ![DataSketch Jaccard Similarity Heatmap](files/text_similarity/DataSketch%20Jaccard%20Similarity%20Heatmap.png)

---

#### 6. Jaccard Similarity Distribution

The histogram demonstrates the frequency distribution of similarity scores:

- **Naïve** captures a broader distribution.
- **LSH and DataSketch** concentrate on higher similarity scores, reflecting their locality-sensitive nature.

![Jaccard Similarity Distribution Across Methods](files/text_similarity/Jaccard%20Similarity%20Distribution%20Accross%20Methods.png)

---

### Summary

- **Naïve (Brute-Force)**: Provides exhaustive comparisons but is computationally expensive.
- **LSH**: Optimized for recall, ensuring most similar pairs are detected but at the cost of precision.
- **DataSketch**: Balances precision, recall, and F1-score, making it a robust alternative to brute-force.

## Execution Time and Performance Comparison

This section compares the execution times and results of the similarity analysis performed using the three techniques:
LSH, Naïve (brute-force), and DataSketch. Key steps and timings for each method are outlined below.

---

#### **1. LSH Technique**

- **Parameter Tuning**: S-curve analysis was performed to optimize the parameters.
    - **Optimal Parameters**: \( r = 16 \), \( b = 20 \) (highest slope on S-curve).
- **Steps**:
    1. Generating MinHash signatures.
        - Processed 4000 sets in approximately 5 minutes.
    2. Indexing MinHash signatures.
        - Completed in **0.40 seconds**.
    3. Finding near-duplicates.
        - Detection completed in **0.49 seconds**.

- **Total Execution Time**: **304.19 seconds** (~5 minutes).
- **Results**:
    - Number of near duplicates found: **748**.
    - Results saved to: `data/processed/near_duplicates_lsh.csv`.

---

#### **2. Naïve (Brute-Force) Technique**

- **Steps**:
    1. Comparing all pairs of shingles.
        - Completed 7,998,000 comparisons in **104.36 seconds** (~1.7 minutes).

- **Total Execution Time**: **104.36 seconds**.
- **Results**:
    - Number of near duplicates found: **436**.
    - Results saved to: `data/processed/near_duplicates_naive.csv`.

---

#### **3. DataSketch Technique**

- **Steps**:
    1. Adding shingles to LSH.
        - Processed 4000 sets in **17 seconds**.
    2. Finding and saving near duplicates.
        - Completed similarity analysis in **18.10 seconds**.

- **Total Execution Time**: **18.10 seconds**.
- **Results**:
    - Number of near duplicates found: **714**.
    - Results saved to: `data/processed/near_duplicates_data_sketch.csv`.

---

#### **Comparison Summary**

| Technique      | Number of Hashes | Total Time (s) | Near Duplicates Found | Notes                                                                          |
|----------------|------------------|----------------|-----------------------|--------------------------------------------------------------------------------|
| **LSH**        | **320**          | 350.47         | 752                   | Faster execution with slightly fewer duplicates; parameters: \( r=16, b=20 \). |
| **LSH**        | **1000**         | 921.27         | 756                   | Slower due to increased computational cost; parameters: \( r=20, b=50 \).      |
| **DataSketch** | **320**          | 18.10          | 714                   | Fastest with balanced results.                                                 |
| **DataSketch** | **1000**         | 39.09          | 684                   | Fastest and highly efficient, with balanced precision and recall.              |
| **Naïve**      | N/A              | 102.16         | 436                   | Exhaustive comparison; efficient for small datasets but scales poorly.         |

---

### Observations

1. **Effect of Number of Hashes on LSH**:
    - Increasing the number of hashes from **320** to **1000** significantly increased the total runtime from **350.47
      seconds** to **921.27 seconds**.
    - The number of near duplicates detected increased slightly (752 to 756), indicating improved accuracy but
      diminishing returns as the computational cost rose significantly.

2. **Execution Time Across Techniques**:
    - Naïve remained faster than both LSH configurations due to its simplicity but lacks scalability for larger
      datasets.
    - DataSketch continues to be the fastest technique, unaffected by the number of hashes since it uses approximate
      methods to achieve balance between speed and accuracy.

3. **Near-Duplicates Detected**:
    - LSH with **1000 hashes** achieved the highest accuracy but at a considerable computational expense.
    - LSH with **320 hashes** offers a more practical balance, achieving nearly the same accuracy as the 1000-hash case
      but with significantly reduced runtime.
    - Naïve detected the least duplicates, showing limitations in recall compared to the hashing-based techniques.
    - DataSketch detected fewer duplicates than LSH, reflecting its trade-off between speed and recall.

---

### Recommendations

- **320 Hashes for LSH**: Use this configuration for a practical trade-off between runtime and accuracy, especially for
  large datasets.
- **1000 Hashes for LSH**: Suitable only when maximum accuracy is required, and computational resources are not a
  concern.
- **DataSketch**: Best suited for scenarios where speed is critical, and slight reductions in recall are acceptable.
- **Naïve**: Only suitable for smaller datasets due to its poor scalability.