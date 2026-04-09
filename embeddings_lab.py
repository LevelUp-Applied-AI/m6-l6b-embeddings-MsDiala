"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on a corpus of climate articles.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Args:
        texts: List of strings (documents).

    Returns:
        Tuple of (tfidf_matrix, vectorizer) where tfidf_matrix is a sparse
        matrix of shape (n_texts, vocab_size) and vectorizer is the fitted
        TfidfVectorizer instance.
    """
    # TODO: Create a TfidfVectorizer, fit-transform on texts, and return
    #       both the matrix and the vectorizer
    pass


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Args:
        tfidf_matrix: Sparse matrix from build_tfidf().

    Returns:
        numpy array of shape (n, n) with pairwise cosine similarities.
    """
    # TODO: Use sklearn's cosine_similarity to compute the pairwise matrix
    pass


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Args:
        filepath: Path to the GloVe text file.

    Returns:
        Dictionary mapping each word (str) to its embedding (numpy array).
    """
    # TODO: Read the file line by line, parse word and vector, store in dict
    pass


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Average the GloVe vectors of all words in the text that are in the
    vocabulary. Skip out-of-vocabulary (OOV) words.

    Args:
        text: Input string.
        embeddings: Dictionary mapping words to numpy arrays (from load_glove).

    Returns:
        numpy array of shape (50,) — the average embedding. If no words are
        found in the vocabulary, return a zero vector of shape (50,).
    """
    # TODO: Tokenize (lowercase, split), look up each word in embeddings,
    #       average the vectors found, handle the all-OOV case
    pass


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Tokenize the text, pass through the model, and mean-pool the
    last hidden state across all tokens to produce a single vector.

    Args:
        text: Input string.
        tokenizer: Hugging Face tokenizer (e.g., DistilBertTokenizer).
        model: Hugging Face model (e.g., DistilBertModel).

    Returns:
        numpy array of shape (768,) — the mean-pooled embedding.
    """
    # TODO: Tokenize with padding/truncation, run model forward pass,
    #       extract last_hidden_state, mean-pool across token dimension
    pass


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query text, find the top-3 most similar texts according to
    each representation method.

    Args:
        texts: List of document strings.
        queries: List of query strings (subset of texts or new queries).
        tfidf_sim: Precomputed TF-IDF similarity matrix (n x n).
        glove_embeddings: Dictionary from load_glove().
        bert_model: Hugging Face DistilBERT model.
        bert_tokenizer: Hugging Face DistilBERT tokenizer.

    Returns:
        Dictionary with structure:
        {
            query_text: {
                'tfidf': [(text, score), ...],   # top-3
                'glove': [(text, score), ...],   # top-3
                'bert':  [(text, score), ...],   # top-3
            }
        }
    """
    # TODO: For each query, compute similarity to all texts using each method,
    #       rank by similarity, and return top-3 for each method
    pass


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, AutoModel

    # Load data
    df = pd.read_csv("data/climate_articles.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_5k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare
    if result and glove and tfidf_sim is not None:
        queries = texts[:5]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in list(comparison.keys())[:2]:
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")
