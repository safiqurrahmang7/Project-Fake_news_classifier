import pandas as pd
from abc import ABC, abstractmethod


class handle_missing_values(ABC):

    @abstractmethod
    def handle(df:pd.DataFrame):
        pass

class drop_missing_values(handle_missing_values):

    def handle(df:pd.DataFrame):

        # Drop missing values
        df= df.dropna(axis=0)
        return df
    
class missing_values_handler:
    
    def __init__(self,handler = handle_missing_values):
        
        self.handler = handler

    def set_handler(self,handler = handle_missing_values):

        self.handler = handler

    def apply_handler(self,df:pd.DataFrame):

        return self.handler.handle(df)