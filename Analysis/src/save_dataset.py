import pandas as pd
from abc import ABC, abstractmethod

# Abstract class defining the interface for saving datasets
class DatasetSaver(ABC):

    @abstractmethod
    def save(self, df: pd.DataFrame, path: str):
        pass

# Concrete class for saving a dataset as a CSV file
class SaveCSV(DatasetSaver):

    def save(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)

# Main class that uses the strategy pattern for saving datasets
class DatasetSaverStrategy:

    def __init__(self, saver: DatasetSaver = None):
        # Set a default saver if no saver is provided
        if saver is None:
            self.saver = SaveCSV()  # Default to SaveCSV if no saver is provided
        else:
            self.saver = saver

    def set_saver(self, saver: DatasetSaver):
        # Set a new saver strategy
        self.saver = saver

    def apply_saver(self, df: pd.DataFrame, path: str):
        # Use the selected saver strategy to save the dataset
        self.saver.save(df, path)
