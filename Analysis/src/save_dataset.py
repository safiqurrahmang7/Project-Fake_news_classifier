import pandas as pd
from abc import ABC, abstractmethod

class dataset_saver(ABC):

    @abstractmethod
    def save(df:pd.DataFrame):
        pass

class save_csv(dataset_saver):

    def save(df:pd.DataFrame, path:str):
        df.to_csv(path, index=False)    

class dataset_saver:

    def __init__(self,saver = dataset_saver):
        
        self.saver = saver
    
    def set_saver(self,saver = dataset_saver):

        self.saver = saver

    def apply_saver(self,df:pd.DataFrame, path:str):

        self.saver.save(df, path)