import pytest
from src.model import build_model
from tensorflow.keras.models import Model

def test_build_model():
    # Test if the model is built correctly
    model = build_model(input_length=100)
    
    # Check if the model is an instance of the Keras Model class
    assert isinstance(model, Model), "The model is not a Keras Model"
    
    # Check if the model has the correct input and output shapes
    assert model.input_shape == (None, 100), "Model input shape is incorrect"
    assert model.output_shape == (None, 1), "Model output shape is incorrect"
