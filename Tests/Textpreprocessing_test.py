import unittest
from unittest.mock import patch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TextPreprocessing.src.text_preprocessing import text_preprocessing_spacy, text_processor

class TestTextProcessor(unittest.TestCase):

    def setUp(self):
        # This will run before each test
        self.processor = text_preprocessing_spacy()
        self.text = "This is a sample text with numbers 1234 and special characters!!!"

    def test_process_method(self):
        # Test the text preprocessing logic
        
        processed_text = self.processor.process(self.text)
        
        # Assert that the processed text does not contain any numbers or special characters
        self.assertNotIn('!', processed_text)
        
        # Assert that the processed text is lowercased and lemmatized
        self.assertTrue(processed_text.islower())
        self.assertIn('sample', processed_text)
        self.assertIn('text', processed_text)
            
    def test_invalid_processor(self):
        # Test if invalid processor handling works, assuming we want to handle it.
        with self.assertRaises(TypeError):
            invalid_processor = text_processor(processor=None)
            invalid_processor.apply_processor(self.text)


if __name__ == '__main__':
    unittest.main()
