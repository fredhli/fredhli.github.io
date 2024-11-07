---
hide:
  - navigation
---
# **Sample Code 2**
[Return to Sample Codes](../sample_codes.md)

## **TF-IDF Matching**

This Python code demonstrates text similarity matching using TF-IDF (*Term Frequency-Inverse Document Frequency*) vectorization and cosine similarity. The code implements efficient blocking and matching functions for comparing large sets of company names or similar text data.

- **Text Processing**: N-gram generation and tokenization
- **Similarity Calculation**: Optimized cosine similarity using sparse matrices
- **TF-IDF Vectorization**: Document feature extraction
- **Efficient Blocking**: Strategic data partitioning for large datasets


## **Code**
```python
import os
import re
import pandas as pd
import numpy as np
import sparse_dot_topn.sparse_dot_topn as ct   

from os.path import join
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer  
from scipy.sparse import csr_matrix


# Text Processing Functions
def ngrams(string, n=3):
    """Generate n-grams from string after removing specific characters."""
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

# Similarity Calculation Functions
def cos_sim_top(matrix_a, matrix_b, n_top_matches, similarity_threshold=0):
    """
    Calculate cosine similarity between two sparse matrices efficiently and return top matches.
    
    Args:
        matrix_a (sparse matrix): First input sparse matrix
        matrix_b (sparse matrix): Second input sparse matrix
        n_top_matches (int): Number of top matches to return
        similarity_threshold (float): Minimum similarity threshold (default: 0)
        
    Returns:
        scipy.sparse.csr_matrix: Sparse matrix containing top similarity scores
    """
    # Convert input matrices to CSR format for efficient computation
    matrix_a = matrix_a.tocsr()
    matrix_b = matrix_b.tocsr()
    
    # Get matrix dimensions
    rows_a, _ = matrix_a.shape
    _, cols_b = matrix_b.shape
    
    # Initialize arrays for result storage
    index_dtype = np.int32
    max_nonzero = rows_a * n_top_matches
    
    result_indptr = np.zeros(rows_a + 1, dtype=index_dtype)
    result_indices = np.zeros(max_nonzero, dtype=index_dtype)
    result_data = np.zeros(max_nonzero, dtype=matrix_a.dtype)

    # Perform sparse dot product using compiled function
    ct.sparse_dot_topn(
        rows_a, cols_b,
        np.asarray(matrix_a.indptr, dtype=index_dtype),
        np.asarray(matrix_a.indices, dtype=index_dtype),
        matrix_a.data,
        np.asarray(matrix_b.indptr, dtype=index_dtype),
        np.asarray(matrix_b.indices, dtype=index_dtype),
        matrix_b.data,
        n_top_matches,
        similarity_threshold,
        result_indptr, result_indices, result_data
    )

    # Return result as CSR matrix
    return csr_matrix((result_data, result_indices, result_indptr), 
                     shape=(rows_a, cols_b))

def get_matches_dataframe(similarity_matrix, names_a, names_b, limit=100):
    """
    Convert sparse similarity matrix to DataFrame of matches.
    
    Args:
        similarity_matrix (sparse matrix): Matrix of similarity scores
        names_a (list): List of names from first dataset
        names_b (list): List of names from second dataset
        limit (int): Maximum number of matches to return
        
    Returns:
        pandas.DataFrame: DataFrame containing match pairs and similarity scores
    """
    # Get non-zero elements
    row_indices, col_indices = similarity_matrix.nonzero()
    
    # Determine number of matches to process
    match_count = min(limit, col_indices.size) if limit else col_indices.size
    
    # Initialize arrays
    source_names = np.empty([match_count], dtype=object)
    target_names = np.empty([match_count], dtype=object)
    scores = np.zeros(match_count)
    
    # Populate match data
    for idx in range(match_count):
        source_names[idx] = names_a[row_indices[idx]]
        target_names[idx] = names_b[col_indices[idx]]
        scores[idx] = similarity_matrix.data[idx]
    
    # Create and return DataFrame
    return pd.DataFrame({
        'left_side': source_names,
        'right_side': target_names,
        'similarity': scores
    })

# Main Blocking and Matching Functions
def blocking_tf_idf(df_source, df_target, block_vars, block_values, name_column, source_id, target_id,
                    tfidf_matrix, vectorizer_type='word', top_matches=100, threshold=0.9, debug=0):
    """
    Perform TF-IDF based blocking and matching on data blocks.
    
    Args:
        df_source (DataFrame): Source dataset block
        df_target (DataFrame): Target dataset block
        block_vars (list): Variables used for blocking
        block_values (list): Values for block variables
        name_column (str): Column name containing company names
        source_id (str): ID column name in source dataset
        target_id (str): ID column name in target dataset
        tfidf_matrix (sparse matrix): Pre-computed TF-IDF matrix
        vectorizer_type (str): Type of vectorizer ('word' or 'char')
        top_matches (int): Number of top matches to consider
        threshold (float): Similarity threshold for matching
        debug (int): Debug level flag
    
    Returns:
        tuple: (match_rate, block_size, matched_data, combinations, unmatched_data)
    """
    if debug == 1:
        print(f'Source block size: {len(df_source)}')
        print(f'Target block size: {len(df_target)}')

    if len(df_source) > 0 and len(df_target) > 0:
        # Extract relevant data
        source_indices = df_source['temp_nameid'].tolist()
        source_names = df_source[name_column].tolist()
        source_tfidf = tfidf_matrix[source_indices]

        target_indices = df_target['temp_nameid'].tolist()
        target_names = df_target[name_column].tolist()
        target_tfidf = tfidf_matrix[target_indices]

        # Calculate similarities
        similarity_matrix = cos_sim_top(source_tfidf, target_tfidf.transpose(), 
                                            top_matches, threshold)
        matches_df = get_matches_dataframe(similarity_matrix, source_names, target_names, 
                                         similarity_matrix.nonzero()[1].size)

        # Select best matches
        np.random.seed(42)
        matches_df['random'] = np.random.normal(size=len(matches_df))
        best_matches = (matches_df.sort_values(['left_side', 'similarity', 'random'], 
                                            ascending=[False, False, False])
                                .drop_duplicates(['left_side'], keep='first'))
        
        # Standardize case
        best_matches['left_side'] = best_matches['left_side'].str.upper()
        best_matches['right_side'] = best_matches['right_side'].str.upper()

        # Create final datasets
        matched_records = pd.merge(df_source, best_matches, 
                                left_on=[name_column], right_on=['left_side'], 
                                how='inner')
        matched_combinations = pd.merge(matched_records, 
                                    df_target[[target_id, name_column]], 
                                    left_on=['right_side'], 
                                    right_on=[name_column], 
                                    how='inner')
        
        # Handle modified names
        if name_column.endswith('_mod'):
            matched_records = process_modified_names(matched_records, df_source, 
                                                  df_target, name_column)

        match_rate = len(matched_records) / len(df_source)
        
    else:
        matched_records, matched_combinations = create_empty_blocks(df_source, 
                                                                name_column, 
                                                                source_id, 
                                                                target_id)
        match_rate = np.NaN

    unmatched_records = get_unmatched_records(df_source, matched_records, source_id)
    
    return match_rate, len(df_source), matched_records, matched_combinations, unmatched_records


def run_blocking_and_matching(output_dir, data_dir, source_path, target_path, temp_dir, 
                            blocking_variables, name_column, source_id, target_id, 
                            vectorizer_type='word', top_matches=100, threshold=0.9,
                            additional_columns=[], debug=0):
    """
    Execute blocking and TF-IDF matching process on datasets.
    
    Args:
        output_dir (str): Directory for output files
        data_dir (str): Directory containing data files
        source_path (str): Path to source dataset
        target_path (str): Path to target dataset
        temp_dir (str): Directory for temporary files
        blocking_variables (list): Variables used for blocking
        name_column (str): Column containing names to match
        source_id (str): ID column in source dataset
        target_id (str): ID column in target dataset
        vectorizer_type (str): Type of vectorizer ('word' or 'char')
        top_matches (int): Number of top matches to consider
        threshold (float): Similarity threshold for matching
        additional_columns (list): Extra columns to include
        debug (int): Debug level flag
    
    Returns:
        str: Status indication
    """
    # Load and preprocess datasets
    source_data = import_and_preprocess_pa(source_path)
    target_data = import_and_preprocess_crsp(target_path)
    
    # Initialize result dictionaries
    stats = {}
    counts = {}
    matched_records = {}
    match_combinations = {}
    unmatched_records = {}
    
    # Generate TF-IDF matrix for all records
    tfidf_matrix = create_tfidf_matrix(source_data, target_data, name_column, vectorizer_type)
    
    # Process data blocks and collect results
    process_blocks(source_data, target_data, blocking_variables, name_column, 
                  source_id, target_id, tfidf_matrix, vectorizer_type,
                  top_matches, threshold, debug, stats, counts, 
                  matched_records, match_combinations, unmatched_records)
    
    # Compile and format results
    summary_statistics = format_statistics(stats, counts)
    final_matches = pd.concat(matched_records.values())
    final_combinations = format_combinations(match_combinations, source_data, 
                                          target_data, source_id, target_id)
    
    # Save processed results
    save_results(output_dir, blocking_variables, vectorizer_type, final_combinations)
    
    return 'completed'
```
[Return to Sample Codes](../sample_codes.md)