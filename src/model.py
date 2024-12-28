import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_length, embedding_dim=100):
    """Build the deep learning model."""
    model = models.Sequential([
        layers.Embedding(input_dim=20000, output_dim=embedding_dim, input_length=input_length),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (fake or real news)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example Usage
# model = build_model(input_length=100)
# model.summary()
