import pytest
import numpy as np
from src.feature_engineering import extract_features, word_embeddings

def test_extract_features():
    # Test if features are extracted correctly
    X = np.array([[1, 2, 3], [4, 5, 6]])  # Dummy data
    X_features = extract_features(X)
    assert X_features.shape == X.shape, "Feature extraction returned incorrect shape"

def test_word_embeddings():
    # Test if word embeddings are correctly applied
    X = np.array([[1, 2, 3], [4, 5, 6]])  # Dummy data
    tokenizer = None  # Replace with actual tokenizer if needed
    X_embedded = word_embeddings(X, tokenizer)
    assert X_embedded.shape == X.shape, "Embedding returned incorrect shape"
