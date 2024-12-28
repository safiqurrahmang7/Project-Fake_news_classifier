import pytest
from src.preprocessing import preprocess_data
import numpy as np

def test_preprocess_data():
    # Test if the data preprocessing function works correctly
    df = pd.read_csv('data/processed/train_data.csv')
    X_train, X_test, y_train, y_test, tokenizer, label_encoder = preprocess_data(df)
    
    # Check if the shapes of X_train and X_test are correct
    assert X_train.shape[0] == len(y_train), "X_train and y_train sizes don't match"
    assert X_test.shape[0] == len(y_test), "X_test and y_test sizes don't match"
    
    # Check if tokenizer and label_encoder are not None
    assert tokenizer is not None, "Tokenizer is None"
    assert label_encoder is not None, "LabelEncoder is None"
    
    # Check if the labels are encoded correctly (i.e., they should be integers)
    assert isinstance(y_train[0], np.int32), "Labels are not encoded as integers"
