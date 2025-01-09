import spacy
import re
from abc import ABC, abstractmethod

class text_preprocessing(ABC):
    @abstractmethod
    def process():
        pass

class text_preprocessing_spacy(text_preprocessing):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=["tok2vec", "tagger", "parser", "attribute_ruler"])

    def process(self, text):

        # Removed special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Removed numbers
        lowered_text = text.lower()

        # Tokenized the text
        doc = self.nlp(lowered_text)

        # Removed stop words and punctuation
        lemmatized = [token.lemma_ for token in doc if  not token.is_punct]

        return ' '.join(lemmatized)
    

class text_processor:

    def __init__(self,processor = text_preprocessing):
        
        self.processor = processor()
    
    def apply_processor(self, text):
        return self.processor.process(text)
    
if __name__ == '__main__':

    preprocessor = text_processor(text_preprocessing_spacy)
    expected = preprocessor.apply_processor('"This is a sample text with numbers 1234 and special characters!!!"')
    print(expected)