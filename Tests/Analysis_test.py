import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Analysis.src.handling_duplicates import drop_duplicates, duplicate_handler
from Analysis.src.handling_missing_values import drop_missing_values, missing_values_handler
from Analysis.src.eda_plots import plot_duplicates, plot_missing_values, plot_wordcloud, plot_distribution, plotter
from Analysis.src.save_dataset import DatasetSaverStrategy,SaveCSV, DatasetSaver
class TestDuplicateHandler(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = {
            'text': ['news1', 'news2', 'news1', 'news3', 'news2'],
            'value': [1, 2, 1, 3, 2]
        }
        self.df = pd.DataFrame(self.data)
        self.handler = duplicate_handler(handler=drop_duplicates)

    def test_drop_duplicates(self):
        # Apply the duplicate handler
        result_df = self.handler.apply_handler(self.df)
        
        # Expected DataFrame after dropping duplicates
        expected_data = {
            'text': ['news1', 'news2', 'news3'],
            'value': [1, 2, 3]
        }
        expected_df = pd.DataFrame(expected_data)
        
        # Check that the resulting DataFrame matches the expected one
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

    def test_set_handler(self):
        # Check the handler can be updated
        new_handler = drop_duplicates
        self.handler.set_handler(new_handler)
        self.assertEqual(self.handler.handler, new_handler)

from Analysis.src.handling_missing_values import drop_missing_values, missing_values_handler

class TestMissingValuesHandler(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = {
            'col1': [1, 2, None, 4],
            'col2': [None, 2, 3, 4]
        }
        self.df = pd.DataFrame(self.data)
        self.handler = missing_values_handler(handler=drop_missing_values)

    def test_drop_missing_values(self):
        # Apply the missing values handler
        result_df = self.handler.apply_handler(self.df)
        
        # Expected DataFrame after dropping rows with missing values
        expected_data = {
            'col1': [1,4],
            'col2': [3,4]
        }
        expected_df = pd.DataFrame(expected_data)
        expected_df = expected_df.astype(float)
        
        # Assert the row count matches
        self.assertEqual(len(result_df), len(expected_df))

    def test_set_handler(self):
        # Check the handler can be updated
        new_handler = drop_missing_values
        self.handler.set_handler(new_handler)
        self.assertEqual(self.handler.handler, new_handler)

class TestEDAPlots(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.data = {
            'text': ['This is a test.', 'Another test.', 'This is a test.', 'Final test.'],
            'target': [1, 0, 1, 0]
        }
        self.df = pd.DataFrame(self.data)
        self.plotter = plotter(plot=plot_duplicates)  # Default plotter for testing

    @patch('matplotlib.pyplot.show')  # Mock plt.show to prevent actual plotting
    def test_plot_duplicates(self, mock_show):
        # Test plot_duplicates without errors
        try:
            self.plotter.apply_plotter(self.df)
            mock_show.assert_called_once()  # Ensure show was called (indicating plot was generated)
        except Exception as e:
            self.fail(f"plot_duplicates raised an exception: {e}")

    @patch('matplotlib.pyplot.show')  # Mock plt.show to prevent actual plotting
    def test_plot_missing_values(self, mock_show):
        # Test plot_missing_values without errors
        self.plotter.set_plotter(plot_missing_values)
        try:
            self.plotter.apply_plotter(self.df)
            mock_show.assert_called_once()  # Ensure show was called (indicating plot was generated)
        except Exception as e:
            self.fail(f"plot_missing_values raised an exception: {e}")

    @patch('matplotlib.pyplot.show')  # Mock plt.show to prevent actual plotting
    def test_plot_wordcloud(self, mock_show):
        # Test plot_wordcloud without errors
        self.plotter.set_plotter(plot_wordcloud)
        try:
            self.plotter.apply_plotter(self.df)
            mock_show.assert_called_once()  # Ensure show was called (indicating plot was generated)
        except Exception as e:
            self.fail(f"plot_wordcloud raised an exception: {e}")

    @patch('matplotlib.pyplot.show')  # Mock plt.show to prevent actual plotting
    def test_plot_distribution(self, mock_show):
        # Test plot_distribution without errors
        self.plotter.set_plotter(plot_distribution(target='target'))
        try:
            self.plotter.apply_plotter(self.df, column='target')
            mock_show.assert_called_once()  # Ensure show was called (indicating plot was generated)
        except Exception as e:
            self.fail(f"plot_distribution raised an exception: {e}")


class TestDatasetSaverStrategy(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.data = {'Name': ['John', 'Jane', 'Jack'],
                     'Age': [28, 34, 29]}
        self.df = pd.DataFrame(self.data)

    @patch('Analysis.src.save_dataset.SaveCSV.save')
    def test_default_saver_csv(self, mock_save):
        # Initialize DatasetSaverStrategy with default (SaveCSV)
        saver_strategy = DatasetSaverStrategy(SaveCSV)

        # Apply saver and check if SaveCSV.save is called
        saver_strategy.apply_saver(self.df, 'test.csv')

        # Assert that the save method of SaveCSV is called with expected arguments
        mock_save.assert_called_once_with(self.df, 'test.csv')

    @unittest.SkipTest
    @patch('Analysis.src.save_dataset.SaveExcel.save')
    def test_custom_saver_excel(self, mock_save):
        # Define a custom SaveExcel class
        class SaveExcel(DatasetSaver):
            def save(self, df: pd.DataFrame, path: str):
                # Simulate the saving process
                pass

        # Initialize DatasetSaverStrategy and set custom saver
        saver_strategy = DatasetSaverStrategy(saver=SaveExcel)
        saver_strategy.apply_saver(self.df, 'test.xlsx')

        # Assert that the save method of SaveExcel is called
        mock_save.assert_called_once_with(self.df, 'test.xlsx')

    def test_apply_saver_with_empty_dataframe(self):
        # Initialize DatasetSaverStrategy with default (SaveCSV)
        saver_strategy = DatasetSaverStrategy(SaveCSV)

        # Create an empty DataFrame
        empty_df = pd.DataFrame()

        # Use patch to mock the file saving process
        with patch('Analysis.src.save_dataset.SaveCSV.save') as mock_save:
            saver_strategy.apply_saver(empty_df, 'empty_test.csv')
            mock_save.assert_called_once_with(empty_df, 'empty_test.csv')

    def test_invalid_path(self):
        # Initialize DatasetSaverStrategy with default (SaveCSV)
        saver_strategy = DatasetSaverStrategy()

        # Simulate an invalid path using unittest.mock
        with self.assertRaises(OSError):
            saver_strategy.apply_saver(self.df, '/invalid_path/test.csv')

    def test_set_saver_method(self):
        # Initialize with default saver (SaveCSV)
        saver_strategy = DatasetSaverStrategy()

        # Change the saver to SaveCSV (default) again
        saver_strategy.set_saver(SaveCSV)
        with patch('Analysis.src.save_dataset.SaveCSV.save') as mock_save:
            saver_strategy.apply_saver(self.df, 'test_set.csv')
            mock_save.assert_called_once_with(self.df, 'test_set.csv')

    def tearDown(self):
        # Clean up any files if needed (e.g., delete created test files)
        if os.path.exists('test.csv'):
            os.remove('test.csv')
        if os.path.exists('test.xlsx'):
            os.remove('test.xlsx')
        if os.path.exists('empty_test.csv'):
            os.remove('empty_test.csv')

if __name__ == '__main__':
    unittest.main()