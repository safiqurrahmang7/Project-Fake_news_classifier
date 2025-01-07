import pandas as pd
import numpy as np
import spacy
import re
from abc import ABC, abstractmethod

class text_preprocessing(ABC):
    @abstractmethod
    def process():
        pass

class text_preprocessing_spacy(text_preprocessing):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "ner"])

    def process(self, text):

        # Removed special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Removed numbers
        lowered_text = text.lower()

        # Tokenized the text
        doc = self.nlp(lowered_text)

        # Removed stop words and punctuation
        lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

        return ' '.join(lemmatized)
    

class text_processor:

    def __init__(self,processor = text_preprocessing):
        
        self.processor = processor()
    
    def apply_processor(self, text):
        return self.processor.process(text)