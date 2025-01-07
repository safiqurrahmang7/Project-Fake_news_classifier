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

    def __init__(self, max_len = 1000,column = 'text'):
        self.max_len = max_len
        self.tokenizer = Tokenizer()
        self.column = column

    def engineering(self, df):
        self.tokenizer.fit_on_texts(df[self.column])
        sequences = self.tokenizer.texts_to_sequences(df[self.column])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        return padded_sequences
    
class feature_engineer:

    def __init__(self, engineer = feature_engineering):

        self.engineer = engineer()

    def apply_engineering(self, df):

        return self.engineer.engineering(df)