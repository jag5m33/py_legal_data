'''
TF-IDF Notes: Sublinear TF Scaling (1 + log(count))

- Core Purpose
    - Reduces the impact of repetition:** Prevents a word repeated 10 times from carrying 10 times the statistical weight. 
    - Implements diminishing returns:** As word counts increase, their added importance slows down dramatically.

- Why the "+ 1" is Mandatory (The Math)
    - The Problem: The logarithm of 1 is exactly 0 (\(\log(1) = 0\)). 
    - The Risk: Without the `+1`, any word appearing exactly **once** in a document would get a term frequency (TF) score of 0. Because TF-IDF multiplies TF × IDF, a 0 score completely deletes that word's weight from the dataset.
    - The Solution: Adding `+1` establishes a baseline weight of **1.0** for single-appearance words while preserving the logarithmic curve for higher counts.

- Practical Impact Breakdown (Using Log Base 10)
    - 1 appearance: \(1 + \log(1) \rightarrow 1 + 0 = \mathbf{1.0}\) (Retains baseline value)
    - 10 appearances: \(1 + \log(10) \rightarrow 1 + 1 = \mathbf{2.0}\) (Only twice as heavy as 1 word)
    - 100 appearances: \(1 + \log(100) \rightarrow 1 + 2 = \mathbf{3.0}\) (Only three times as heavy as 1 word)

- Python/Scikit-Learn Implementation
    - Trigger this behavior by setting the parameter **`sublinear_tf=True`** inside `TfidfVectorizer`.
    - Note: Scikit-learn utilizes the natural log (\(\ln\)) rather than base 10, but the underlying mathematical logic and necessity for the `+1` remain identical.
'''

from sklearn.feature_extraction.text import TfidfVectorizer

def tfid(stop_words_set):
    vectorizer = TfidfVectorizer(stop_words= (stop_words_set), 
                                 lowercase = True, 
                                 ngram_range = (1,2), 
                                 min_df = 4, 
                                 sublinear_tf = True, 
                                 max_df = 0.9, 
                                 norm = 'l2') 
    return vectorizer

# l2 normalisation - sum of squares, normalise (divide by length)