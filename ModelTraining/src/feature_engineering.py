import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from abc import ABC, abstractmethod


class feature_engineering(ABC):
    @abstractmethod
    def engineering():
        pass

class text_feature_engineering(feature_engineering):

    def __init__(self, max_len = 1000, column = 'text'):

        self.max_len = max_len
        self.column = column
    
    def engineering(self, df):
        df[self.column] = df[self.column].astype(str)
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(df[self.column])
        sequences = tokenizer.texts_to_sequences(df[self.column])
        padded_text = pad_sequences(sequences,maxlen = 1000, padding='post', truncating='post')
        return padded_text
    
class feature_engineer:

    def __init__(self, engineer = feature_engineering):

        self.engineer = engineer

    def apply_engineering(self, df):

        return self.engineer.engineering(df)