from sklearn.model_selection import train_test_split
import pandas as pd
from abc import ABC, abstractmethod

class data_splitter(ABC):
    @abstractmethod
    def split_data():
        pass

class train_test_splitter(data_splitter):
    def __init__(self, test_size = 0.2, random_state = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, feature,target):
        X = feature
        y = target.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
    
class data_split:

    def __init__(self, splitter = data_splitter):
        self.splitter = splitter()

    def split_data(self, feature, target):
        return self.splitter.split_data(feature, target)