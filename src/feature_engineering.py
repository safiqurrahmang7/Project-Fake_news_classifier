# Import necessary libraries
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

def prepare_text_data(df, text_column='text', max_features=5000, max_length=100):
    """
    Preprocess the text data by tokenizing and padding the sequences.
    
    Args:
    - df: DataFrame containing the text data.
    - text_column: The column containing the text (default is 'text').
    - max_features: Maximum number of words to keep in the tokenizer (default is 5000).
    - max_length: Maximum length of the sequences (default is 100).
    
    Returns:
    - X: Tokenized and padded text data.
    - tokenizer: The fitted tokenizer used to convert text to sequences.
    """
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df[text_column])
    sequences = tokenizer.texts_to_sequences(df[text_column])
    X = pad_sequences(sequences, maxlen=max_length)
    
    return X, tokenizer

