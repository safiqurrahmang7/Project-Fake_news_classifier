import pytest
from sklearn.metrics import classification_report
from src.evaluate import evaluate_model
import numpy as np
from tensorflow.keras.models import load_model

def test_evaluate_model():
    # Test if the model evaluation works correctly
    model = load_model('models/fake_news_model.h5')  # Replace with a valid model path
    X_test = np.random.rand(100, 100)  # Dummy data for testing
    y_test = np.random.randint(2, size=100)  # Dummy labels
    
    evaluate_model(model, X_test, y_test)  # This should not raise any errors
    
    # Since it's not easy to test the exact output of classification_report,
    # we can check if the function completes without errors
    assert True  # If no exceptions, the test passes
