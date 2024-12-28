import pytest
from tensorflow.keras.models import Sequential
from src.save_model import save_trained_model
import os

def test_save_trained_model():
    # Create a simple model for testing
    model = Sequential()
    model.add(Dense(10, input_dim=100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model_path = 'models/test_model.h5'
    save_trained_model(model, model_path)
    
    # Test if the model was saved correctly
    assert os.path.exists(model_path), "Model was not saved correctly"
    
    # Clean up the saved model file
    os.remove(model_path)
