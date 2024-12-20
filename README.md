
# Text Similarity and Clustering

The Text Similarity and Clustering project implements efficient methods like shingling, minwise hashing, and LSH to
detect near-duplicate Amazon product descriptions and compares their performance to brute-force and library-based
approaches. It also performs clustering on the California Housing Prices dataset, analyzing the impact of feature
engineering on clustering quality and efficiency. Deliverables include code, reports, and visualizations summarizing
findings and results.

# Table of Contents

- [Text Similarity](#text-similarity)
    - [Approach Overview](#approach-overview)
    - [LSH Pipeline](#lsh-pipeline)
    - [Analyzing Amazon Product Descriptions](#analyzing-amazon-product-descriptions)
    - [Preprocessing Descriptions for Near-Duplicate Search](#preprocessing-descriptions-for-near-duplicate-search)
        - [Steps in Preprocessing](#steps-in-preprocessing)
            - [1. Multi-Word Term Preservation](#1-multi-word-term-preservation)
            - [2. Tokenization](#2-tokenization)
            - [3. Punctuation and Symbol Removal](#3-punctuation-and-symbol-removal)
            - [4. Handling Joined Terms](#4-handling-joined-terms)
            - [5. Stopword Removal](#5-stopword-removal)
            - [6. Token Processing (Stemming and Lemmatization)](#6-token-processing-stemming-and-lemmatization)
            - [7. Multi-Word Term Restoration](#7-multi-word-term-restoration)
        - [Example Preprocessing](#example-preprocessing)
    - [Shingling Process](#shingling-process)
        - [Implementation Details](#implementation-details)
    - [Min-Hashing Process](#min-hashing-process)
        - [Implementation Details](#implementation-details-1)
    - [Finding Near-Duplicates (using LSH)](#finding-near-duplicates-using-lsh)
        - [Implementation Details](#implementation-details-2)
  - [S-Curve Analysis for Parameter Optimization](#s-curve-analysis-for-parameter-optimization)
      - [Purpose](#purpose)
      - [Process](#process)
      - [Results](#results)
      - [Summary](#summary)
  - [Advantages of LSH](#advantages-of-lsh)
  - [Example Workflow with LSH](#example-workflow-with-lsh)
      - [Step 1: Parameter Optimization with S-Curve Analysis](#step-1-parameter-optimization-with-s-curve-analysis)
      - [Step 2: Generating Min-Hash Signatures](#step-2-generating-min-hash-signatures)
      - [Step 3: Indexing Min-Hash Signatures](#step-3-indexing-min-hash-signatures)
      - [Step 4: Identifying Near-Duplicates](#step-4-identifying-near-duplicates)
  - [Comparison of Naïve, LSH, and DataSketch Methods for Text Similarity](#comparison-of-naïve-lsh-and-datasketch-methods-for-text-similarity)
      - [1. Venn Diagram: Overlap in Detected Similar Pairs](#1-venn-diagram-overlap-in-detected-similar-pairs)
      - [2. Jaccard Similarity Box Plot](#2-jaccard-similarity-box-plot)
      - [3. Precision, Recall, and F1-Score Comparison](#3-precision-recall-and-f1-score-comparison)
      - [4. Cumulative Distribution of Jaccard Similarities](#4-cumulative-distribution-of-jaccard-similarities)
      - [5. Jaccard Similarity Heatmaps](#5-jaccard-similarity-heatmaps)
      - [6. Jaccard Similarity Distribution](#6-jaccard-similarity-distribution)
  - [Summary](#summary-1)
  - [Execution Time and Performance Comparison](#execution-time-and-performance-comparison)
      - [1. LSH Technique](#1-lsh-technique)
      - [2. Naïve (Brute-Force) Technique](#2-naïve-brute-force-technique)
      - [3. DataSketch Technique](#3-datasketch-technique)
      - [Comparison Summary](#comparison-summary)
      - [Observations](#observations)
      - [Recommendations](#recommendations)

- [Clustering](#clustering)
    - [Introduction](#introduction)
    - [Dataset Overview](#dataset-overview)
    - [Methodology](#methodology)
        - [Clustering Algorithm k-means++](#clustering-algorithm-k-means)
        - [Methods for Determining Optimal Clusters](#methods-for-determining-optimal-clusters)
            - [Silhouette Score](#silhouette-score)
            - [Davies-Bouldin Index](#davies-bouldin-index)
            - [Calinski-Harabasz Index](#calinski-harabasz-index)
            - [Elbow Method](#elbow-method)
    - [Feature Engineering](#feature-engineering)
    - [Clustering on Raw Data](#clustering-on-raw-data)
    - [Clustering on Feature-Engineered Data](#clustering-on-feature-engineered-data)
    - [Comparison of Clustering Results](#comparison-of-clustering-results)
        - [Raw vs Feature-Engineered Data](#raw-vs-feature-engineered-data)
        - [Evaluation Metrics and Insights](#evaluation-metrics-and-insights)
  - [Conclusion and Future Work](#conclusion)

## Text Similarity

This part aims to identify near-duplicate products within a dataset of Amazon product descriptions by implementing a
**nearest-neighbor search** using text-based methods. The task focuses on leveraging techniques such as **shingling**,
**minwise hashing**, and **locality-sensitive hashing (LSH)** to detect similarities in textual content efficiently.

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

When scraping Amazon for PC or laptop listings, the scraped data is saved as a file
named `laptops_results_2024-11-23.tsv` in the `data/raw` directory.
Upon cloning the repository you have to unrar the raw.rar and make sure the file `laptops_results_2024-11-23.tsv` exists
in `data/raw`.

To address the near-duplicate search problem, we perform the search in two ways:

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

#### Why This Approach?
- **Flexibility**: Supports both character-level and token-level shingles, allowing adaptation to the type of input data and the granularity of similarity detection.
- **Compact Representation**: By using sets, duplicate shingles within the same document are avoided, optimizing storage and processing.

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

The S-curve analysis is conducted to optimize the parameters of the LSH algorithm r : rows per band, b :
number of bands) for achieving a balance between false positives and false negatives. This ensures effective similarity
detection while maintaining computational efficiency.

---

#### **Process**

1. **Generating S-Curves**:
    - Multiple S-curves are plotted for various \( r, b \) combinations, illustrating the probability of candidate
      selection as a function of Jaccard similarity.
   - Steeper slopes at the similarity threshold \( s = 0.8 \) indicate better performance in distinguishing similar
      and dissimilar pairs.

2. **Parameter Tuning**:
    - The optimal \( r, b \) combination is identified as the one that maximizes the slope of the S-curve at s =
      0.8 .
    - Two cases were explored:
        - **Moderate Parameters ( r = 16, b = 20 )**: Faster execution with acceptable accuracy.
        - **High Parameters ( r = 20, b = 50 )**: Improved accuracy but slower due to computational overhead.

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
      b=50 \)) is steeper at the threshold, offering better precision, while the blue curve ( r=16, b=20 ) provides
      a more computationally efficient solution with a slightly lower slope.

   ![Difference Between (r=16, b=20) and (r=20, b=50)](files/text_similarity/the%20difference%20between%20different%20(r,b)%20pairs%20-%20(r=16,b=20)%20vs%20(r=20,%20b=50).png)

---

#### **Summary**

- **Moderate Parameters ( r = 16, b = 20 )**:
    - Faster execution, suitable for large datasets where computational cost is a concern.
    - A good trade-off between accuracy and efficiency.

- **High Parameters ( r = 20, b = 50 )**:
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
- The number of hash functions is calculated as \( r x b \), ensuring the signatures align with the chosen
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

The box plot illustrates the distribution of Jaccard similarity scores for the three methods used to detect
near-duplicate documents:

- **Naïve (Brute-force):**
    - The Jaccard similarity scores show a wide range, indicating variability in how well near-duplicates were
      identified.
    - Outliers close to 1.0 represent exact or near-exact matches, while the lower bound highlights some pairs with
      minimum similarity close to the threshold of 0.8.
    - This range reflects the comprehensive but computationally intensive nature of brute-force comparison.

- **LSH (Locality Sensitive Hashing):**
    - The Jaccard similarities are more concentrated around higher values, demonstrating that the algorithm effectively
      captures document pairs with higher similarity.
    - The narrower interquartile range (IQR) and higher median compared to the Naïve method suggest more consistent
      results with fewer extreme variations.

- **DataSketch (LSH Implementation):**
    - Similar to LSH, this method produces a focused range of similarity scores but exhibits slightly more spread than
      LSH, potentially due to differences in implementation or parameter tuning.
    - The performance is comparable to LSH, highlighting its robustness and computational efficiency in identifying
      near-duplicates.

![Jaccard Similarity Box Plot](files/text_similarity/Jaccard%20Similarity%20Box%20Plot.png)

---

#### 3. Precision, Recall, and F1-Score Comparison

The bar plots highlight performance metrics:

- **LSH** achieves higher recall but lower precision than DataSketch.
- **DataSketch** offers balanced metrics with slightly better precision and F1-score.

![LSH and DataSketch Metrics Comparison](files/text_similarity/Comparison%20of%20LSH%20and%20DataSketch%20Metrics.png)

---

#### 4. Cumulative Distribution of Jaccard Similarities

The cumulative distribution plot compares how similarity scores are distributed

![Cumulative Distribution of Jaccard Similarities](files/text_similarity/Cumulative%20Distribution%20of%20Jaccard%20Similarities.png)

---

#### 5. Jaccard Similarity Heatmaps

The heatmaps visualize pairwise Jaccard similarities

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

## Clustering

This project focuses on clustering the **California Housing Prices** dataset using the **k-means++ algorithm**. The
optimal number of clusters is determined and compared using four different methods: Silhouette Score, Davies-Bouldin
Index, Calinski-Harabasz Index, and the Elbow Method. Clustering is performed on both raw and feature-engineered data,
with visualizations and metrics to evaluate the quality of the clusters and the impact of feature engineering.

---

### Introduction

### Dataset Overview

The dataset, derived from the 1990 California census, contains housing information for districts, including geographic,
demographic, and economic features. It includes the following columns:

- **Geographic Features:** `longitude`, `latitude`
- **Demographic Features:** `population`, `households`, `housing_median_age`
- **Economic Features:** `median_income`, `median_house_value`
- **Housing Features:** `total_rooms`, `total_bedrooms`, `ocean_proximity`

This dataset requires preprocessing and is ideal for introducing machine learning concepts due to its manageable size
and straightforward variables.

#### Downloading and Loading the Dataset

The dataset can be downloaded and loaded using the **`main_clustering.py`** script as follows:

1. **Download the Dataset**:
    - Run the script with the `download` action to download the dataset from Kaggle:
      ```bash
      python main_clustering.py download
      ```

2. **Load and Clean the Dataset**:
    - Use the `load` action to load the dataset and clean missing values:
      ```bash
      python main_clustering.py load
      ```

These steps ensure the dataset is available in the specified directory (`data/raw/housing.csv`) and is ready for further
processing and clustering.

### Methodology

#### Clustering Algorithm k-means++

The **k-means++ algorithm** was chosen for clustering due to its efficiency and improved cluster initialization over
standard k-means. It minimizes variance within clusters by iteratively assigning data points to the nearest cluster
centroids and recalculating centroids. The use of k-means++ ensures better initial cluster center selection, reducing
the likelihood of poor convergence.

#### Methods for Determining Optimal Clusters

To determine the optimal number of clusters, four established evaluation methods were used:

##### Silhouette Score

The Silhouette Score evaluates clustering quality by measuring how well-separated clusters are. It considers how close a
data point is to points within its cluster versus points in the nearest neighboring cluster. Scores range from -1 to 1:

- A higher score indicates well-separated and distinct clusters.
- Negative scores suggest incorrect cluster assignment.

**Implementation in Code:**  
The difference in scores between successive clusters (\( k \)) is computed. The sharpest drop in scores, determined by
the minimum difference, identifies the optimal number of clusters. This is calculated as:

```python
score_diffs = np.diff(scores)
sharpest_drop_idx = np.argmin(score_diffs)
optimal_clusters = range_values[sharpest_drop_idx]
```

##### Davies-Bouldin Index

The Davies-Bouldin Index measures clustering quality by assessing intra-cluster compactness and inter-cluster
separation. Lower index values indicate:

- Compact clusters with minimal overlap.
- Better-separated clusters.

**Implementation in Code:**  
Scores are calculated for different \( k \) values. Clusters with \( k > 3 \) are filtered out for higher robustness,
and the minimum score is used to determine the optimal cluster count:

```python
filtered_range_values = [k for k in range_values if k > 3]
filtered_scores = [scores[range_values.index(k)] for k in filtered_range_values]
optimal_clusters = filtered_range_values[filtered_scores.index(min(filtered_scores))]
```

##### Calinski-Harabasz Index

The Calinski-Harabasz Index, also called the Variance Ratio Criterion, evaluates clustering by comparing the variance
within clusters to the variance between cluster centroids. Higher scores indicate better-defined clusters with
significant differences between centroids.

**Implementation in Code:**  
The optimal cluster count corresponds to the \( k \) value yielding the highest score:

```python
optimal_clusters = range_values[scores.index(max(scores))]
```

##### Elbow Method

The Elbow Method uses **inertia**, representing the sum of squared distances of points to their nearest cluster
centroid. Plotting inertia against \( k \) values reveals an "elbow point" where the reduction in inertia slows down,
marking the optimal cluster count.

**Implementation in Code:**  
To objectively identify the elbow point, the "knee detection" technique normalizes scores and calculates perpendicular
distances from a line connecting the first and last inertia values:

```python
normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
distances = []
for i in range(len(normalized_scores)):
    x1, y1 = 0, normalized_scores[0]
    x2, y2 = len(normalized_scores) - 1, normalized_scores[-1]
    xi, yi = i, normalized_scores[i]
    numerator = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances.append(numerator / denominator)
optimal_clusters = range_values[np.argmax(distances)]
```

Each method provides unique insights into clustering quality, ensuring robust identification of the optimal cluster
count.
### Feature Engineering

Feature engineering was applied to enhance the dataset's usability for clustering by transforming and normalizing raw
data. The following techniques were implemented:

- **Handling Missing Data:** Missing values in numeric features were imputed using the median value to ensure a complete
  dataset for clustering.

- **Scaling Numeric Features:** Numeric features such as `total_rooms`, `population`, and `median_income` were
  standardized using `StandardScaler` to normalize their distribution and ensure equal importance during clustering.

- **One-Hot Encoding:** The categorical feature `ocean_proximity` was encoded into binary columns using one-hot encoding
  to allow its integration into the clustering process.

- **Adding Interaction Features:** New features, such as `rooms_per_household` and `bedrooms_per_household`, were
  created to capture additional relationships between existing variables.

- **Discretizing Housing Age:** The `housing_median_age` feature was discretized into bins (
  e.g., `0-20`, `20-40`, `40+`) and one-hot encoded to reduce the effect of outliers.

These engineered features enhanced the dataset's structure, improving the clustering algorithm's ability to
differentiate between groups.

### Clustering on Raw Data

Clustering on the raw dataset was performed using the **k-means++ algorithm**. The optimal number of clusters was
determined using various evaluation methods, including Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index,
and the Elbow Method.

#### Commands:

To run clustering on raw data we have to first `load` the data, pass `clustering_raw` and our desired `score_method`:

```bash
python main_clustering.py load clustering_raw --score_method silhouette
```

```bash
python main_clustering.py load clustering_raw --score_method davies_bouldin
```

```bash
python main_clustering.py load clustering_raw --score_method calinski_harabasz
```

```bash
python main_clustering.py load clustering_raw --score_method elbow
```

### Clustering on Feature-Engineered Data

Clustering was repeated on the feature-engineered dataset to evaluate the impact of preprocessing. Preprocessing steps
included scaling, one-hot encoding, feature creation, and handling missing data.


#### Commands:

Just as we have done for raw data, to run clustering on feature-engineered data we have to first `load` the data,
pass `clustering_engineered` and our desired `score_method`:

```bash
python main_clustering.py load clustering_engineered --score_method silhouette
```

```bash
python main_clustering.py load clustering_engineered --score_method davies_bouldin
```

```bash
python main_clustering.py load clustering_engineered --score_method calinski_harabasz
```

```bash
python main_clustering.py load clustering_engineered --score_method elbow
```

#### Key Visualizations:

| **Method**                   | **Raw Data**                                                                                                              | **Feature-Engineered Data**                                                                                             |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| **Silhouette Method**        | ![Raw Silhouette Method](files/clustering/Raw%20-%20Optimal%20Clusters%20using%20Silhouette%20Method.png)                 | ![FE Silhouette Method](files/clustering/FE%20-%20Optimal%20Clusters%20using%20Silhouette%20Method.png)                 |
| **Optimal Clusters**         | <p align="center"><b>2</b></p>                                                                                            | <p align="center"><b>6</b></p>                                                                                          |
| **Davies-Bouldin Method**    | ![Raw Davies-Bouldin Method](files/clustering/Raw%20-%20Optimal%20Clusters%20using%20Davies%20Bouldin%20Method.png)       | ![FE Davies-Bouldin Method](files/clustering/FE%20-%20Optimal%20Clusters%20using%20Davies%20Bouldin%20Method.png)       |
| **Optimal Clusters**         | <p align="center"><b>5</b></p>                                                                                            | <p align="center"><b>10</b></p>                                                                                         |
| **Calinski-Harabasz Method** | ![Raw Calinski-Harabasz Method](files/clustering/Raw%20-%20Optimal%20Clusters%20using%20Calinski%20Harabasz%20Method.png) | ![FE Calinski-Harabasz Method](files/clustering/FE%20-%20Optimal%20Clusters%20using%20Calinski%20Harabasz%20Method.png) |
| **Optimal Clusters**         | <p align="center"><b>49</b></p>                                                                                           | <p align="center"><b>5</b></p>                                                                                          |
| **Elbow Method**             | ![Raw Elbow Method](files/clustering/Raw%20-%20Optimal%20Clusters%20using%20Elbow%20Method.png)                           | ![FE Elbow Method](files/clustering/FE%20-%20Optimal%20Clusters%20using%20Elbow%20Method.png)                           |
| **Optimal Clusters**         | <p align="center"><b>7</b></p>                                                                                            | <p align="center"><b>10</b></p>                                                                                         |

--- 

| **Visualization Type** | **Raw Data**                                                                                 | **Feature-Engineered Data**                                                                |
|------------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **2D Visualization**   | ![Raw 2D Visualization](files/clustering/Raw%20-%20Visualizing%20Data%20Labels%20-%202D.png) | ![FE 2D Visualization](files/clustering/FE%20-%20Visualizing%20Data%20Labels%20-%202D.png) |
| **3D Visualization**   | ![Raw 3D Visualization](files/clustering/Raw%20-%20Visualizing%20Data%20Labels%20-%203D.png) | ![FE 3D Visualization](files/clustering/FE%20-%20Visualizing%20Data%20Labels%20-%203D.png) |

#### Observations:

- The raw data clustering resulted in less distinct clusters, as reflected by lower silhouette and Calinski-Harabasz
  scores, and fluctuating Davies-Bouldin Index.
- The elbow point was less pronounced, suggesting suboptimal cluster separability.
- The feature-engineered data resulted in more compact and well-separated clusters across all methods.
- The optimal cluster count was more distinct, as seen in the Calinski-Harabasz and Elbow Method results.
- The silhouette score was consistently higher, indicating better cluster quality.

---

### Comparison of Clustering Results

#### Raw vs Feature-Engineered Data

The feature-engineered dataset showed significant improvements across all evaluation methods:

- **Silhouette Score**: Higher scores for feature-engineered data indicate better cluster compactness and separation.
- **Davies-Bouldin Index**: Lower scores for feature-engineered data demonstrate more compact and well-separated
  clusters.
- **Calinski-Harabasz Index**: Higher scores for feature-engineered data reflect better-defined clusters.
- **Elbow Method**: The elbow point was more distinct for the feature-engineered data, suggesting clearer separability.

#### Evaluation Metrics and Insights

| Metric                  | Raw Data              | Feature-Engineered Data |
|-------------------------|-----------------------|-------------------------|
| Silhouette Score        | Lower and less stable | Higher and more stable  |
| Davies-Bouldin Index    | Higher (less compact) | Lower (more compact)    |
| Calinski-Harabasz Index | Less distinct peaks   | More distinct peaks     |
| Elbow Method            | Subtle elbow point    | Pronounced elbow point  |

**Insights:**

- Feature engineering enhanced cluster compactness and separability.
- Preprocessing made clustering results more interpretable and robust.

---

### Conclusion

Clustering results improved significantly after feature engineering. The feature-engineered data produced:

1. Higher silhouette scores, indicating better cluster compactness.
2. Lower Davies-Bouldin Index values, reflecting better-separated clusters.
3. More distinct peaks in the Calinski-Harabasz Index, highlighting optimal cluster numbers.
4. Clearer elbow points, emphasizing better separability.
5. Clustering on raw data, in all the executions was faster (having lower execution time) since it was performed on data
   with lower dimension and less number of data.
 