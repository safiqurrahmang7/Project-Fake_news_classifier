import unittest
import pandas as pd
import numpy as np
from ModelTraining.src.feature_engineering import text_feature_engineering, feature_engineer  # Replace with your actual module name
from ModelTraining.src.model_building import Build_LSTM_model, model_builder
from ModelTraining.src.model_evaluating import model_evaluation, model_evaluator
from tensorflow.keras.models import Sequential
from unittest.mock import Mock
from sklearn.model_selection import train_test_split
from ModelTraining.src.data_splitter import train_test_splitter, data_split  

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.df = pd.DataFrame({
            'text': ['This is a test', 'Another test text', 'Final sample']
        })
        self.expected_shape = (3, 1000)  # 3 rows, maxlen=1000
    
    def test_text_feature_engineering(self):
        # Test the text_feature_engineering class
        text_engineer = text_feature_engineering(column='text')
        padded_text = text_engineer.engineering(self.df)

        # Check the type and shape of the result
        self.assertIsInstance(padded_text, np.ndarray)
        self.assertEqual(padded_text.shape, self.expected_shape)
    
    def test_feature_engineer(self):
        # Test the feature_engineer wrapper class
        text_engineer = text_feature_engineering(column='text')
        feature_eng = feature_engineer(engineer=text_engineer)
        padded_text = feature_eng.apply_engineering(self.df)

        # Check the type and shape of the result
        self.assertIsInstance(padded_text, np.ndarray)
        self.assertEqual(padded_text.shape, self.expected_shape)
    
    def test_empty_dataframe(self):
        # Test behavior with an empty DataFrame
        df_empty = pd.DataFrame({'text': []})
        text_engineer = text_feature_engineering(column='text')
        padded_text = text_engineer.engineering(df_empty)

        # Check if the result is an empty array
        self.assertIsInstance(padded_text, np.ndarray)
        self.assertEqual(padded_text.shape, (0, 1000))
    
    def test_non_string_column(self):
        # Test behavior with a non-string column
        df_mixed = pd.DataFrame({'text': [123, None, 'Valid text']})
        text_engineer = text_feature_engineering(column='text')
        padded_text = text_engineer.engineering(df_mixed)

        # Check if the result is processed correctly
        self.assertIsInstance(padded_text, np.ndarray)
        self.assertEqual(padded_text.shape, (3, 1000))


class TestDataSplitter(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        self.feature = pd.DataFrame({
            'feature1': np.arange(10),
            'feature2': np.arange(10, 20)
        })
        self.target = pd.Series(np.random.randint(0, 2, size=10))

    def test_train_test_splitter(self):
        # Test the train_test_splitter class directly
        splitter = train_test_splitter(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = splitter.split_data(self.feature, self.target)

        # Check shapes of the splits
        self.assertEqual(X_train.shape[0], 7)  # 70% training data
        self.assertEqual(X_test.shape[0], 3)   # 30% test data
        self.assertEqual(y_train.shape[0], 7)  # 70% training target
        self.assertEqual(y_test.shape[0], 3)   # 30% test target

        # Check if the split is consistent with the random_state
        X_train_expected, X_test_expected, y_train_expected, y_test_expected = train_test_split(
            self.feature, self.target.values, test_size=0.3, random_state=42
        )
        pd.testing.assert_frame_equal(X_train, X_train_expected)
        pd.testing.assert_frame_equal(X_test, X_test_expected)
        np.testing.assert_array_equal(y_train, y_train_expected)
        np.testing.assert_array_equal(y_test, y_test_expected)

    def test_data_split_wrapper(self):
        # Test the data_split wrapper class
        splitter = train_test_splitter(test_size=0.2, random_state=42)
        wrapper = data_split(splitter=splitter)
        X_train, X_test, y_train, y_test = wrapper.split_data(self.feature, self.target)

        # Check shapes of the splits


class TestBuildLSTMModel(unittest.TestCase):

    def test_model_builder_wrapper(self):
        # Test the model_builder wrapper class
        lstm_builder = Build_LSTM_model()
        wrapper = model_builder(builder=lstm_builder)
        model = wrapper.apply_model_builder()

        # Ensure the wrapper correctly applies the build_model method
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 7)


class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        # Mock model for testing
        self.mock_model = Mock()
        self.mock_model.predict = Mock(return_value=np.array([[0.7], [0.3], [0.9], [0.2]]))

        # Test input data
        self.X_test = np.random.rand(4, 10)  # 4 samples, 10 features each

    def test_model_evaluation(self):
        # Test the model_evaluation class directly
        evaluator = model_evaluation
        y_pred_classes = evaluator.evaluate(self.mock_model, self.X_test)

        # Expected predictions converted to binary classes
        expected_pred_classes = np.array([[1], [0], [1], [0]])

        # Verify predictions
        np.testing.assert_array_equal(y_pred_classes, expected_pred_classes)

    def test_model_evaluator_wrapper(self):
        # Test the model_evaluator wrapper class
        evaluator = model_evaluation
        evaluator_wrapper = model_evaluator(evaluator=evaluator)
        y_pred_classes = evaluator_wrapper.evaluate_model(self.mock_model, self.X_test)

        # Expected predictions converted to binary classes
        expected_pred_classes = np.array([[1], [0], [1], [0]])

        # Verify predictions
        np.testing.assert_array_equal(y_pred_classes, expected_pred_classes)

    def test_mock_model_called(self):
        # Verify that the mock model's predict method is called
        evaluator = model_evaluation
        evaluator.evaluate(self.mock_model, self.X_test)

        # Assert the predict method was called once with the correct input
        self.mock_model.predict.assert_called_once_with(self.X_test)

if __name__ == '__main__':
    unittest.main()