import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class handle_duplicate(ABC):

    @abstractmethod
    def handle(df:pd.DataFrame):
        pass


class drop_duplicates(handle_duplicate):

    def handle(df:pd.DataFrame):

        # Drop duplicates
        df = df.drop_duplicates(subset = 'text',keep='first')
        return df
    
class duplicate_handler:

    def __init__(self,handler = handle_duplicate):
        
        self.handler = handler

    def set_handler(self,handler = handle_duplicate):

        self.handler = handler

    def apply_handler(self,df:pd.DataFrame):

        return self.handler.handle(df)