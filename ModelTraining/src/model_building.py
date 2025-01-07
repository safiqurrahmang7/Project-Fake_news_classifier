import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from abc import ABC, abstractmethod

class Build_model(ABC):
    @abstractmethod
    def build_model():
        pass

class Build_LSTM_model(Build_model):

    def build_model(self):

        model = Sequential()
        model.add(Embedding(input_dim=10000,output_dim = 256, input_length=200))  # Embedding layer
        model.add(SpatialDropout1D(0.3))
        model.add(LSTM(128, return_sequences=True))  # Bidirectional LSTM layer
        model.add(Dropout(0.3))  # Dropout for regularization
        model.add(LSTM(64))  # Another Bidirectional LSTM layer
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

        # Compiling the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model
    

class model_builder:

    def __init__(self, builder = Build_model):

        self.builder = builder

    def apply_model_builder(self):

        return self.builder.build_model()


    