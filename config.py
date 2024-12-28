import os

# Project root directory (adjust this to the root of your project)
PROJECT_DIR = r'D:\Project-Fake_news_classifier'

# Paths for data
RAW_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'processed')

# Model saving path
MODEL_PATH = os.path.join(PROJECT_DIR, 'models')

# Paths for notebooks
NOTEBOOKS_PATH = os.path.join(PROJECT_DIR, 'notebooks')

# Hyperparameters for the model
MAX_LENGTH = 100  # Maximum sequence length for tokenization
BATCH_SIZE = 32   # Batch size for training
EPOCHS = 10       # Number of epochs for training
LEARNING_RATE = 0.001  # Learning rate for the optimizer

# Path to store trained models
MODEL_SAVE_PATH = os.path.join(MODEL_PATH, 'fake_news_model.h5')

# Path for logs (if using TensorBoard or custom logs)
LOGS_PATH = os.path.join(PROJECT_DIR, 'logs')

# File paths for storing processed data (e.g., training, testing datasets)
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train_data.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'test_data.csv')

# Pre-trained word embeddings (if using embeddings like GloVe)
EMBEDDING_PATH = os.path.join(PROJECT_DIR, 'data', 'embeddings', 'glove.6B.100d.txt')

# Set to True for debugging or experimenting with smaller data
DEBUG_MODE = False

# Set this to True if you want to enable GPU support (if available)
USE_GPU = True

# Paths for test results
TEST_RESULTS_PATH = os.path.join(PROJECT_DIR, 'tests', 'test_results.txt')

# Set up logging configurations
LOGGING_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# Other configuration settings
RANDOM_SEED = 42  # Set random seed for reproducibility
