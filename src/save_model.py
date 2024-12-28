import tensorflow as tf

def save_trained_model(model, model_path):
    """Save the trained model to the specified path."""
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Example Usage
# save_trained_model(model, 'models/fake_news_model.h5')
