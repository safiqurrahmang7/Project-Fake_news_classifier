import pytest
import pandas as pd
from src.eda import load_data, basic_info, plot_class_distribution, missing_values

def test_load_data():
    # Test if the data is loaded correctly
    df = load_data('data/processed/train_data.csv')
    assert isinstance(df, pd.DataFrame), "Data is not loaded as a DataFrame"

def test_basic_info():
    # Test if basic info is printed correctly
    df = pd.read_csv('data/processed/train_data.csv')
    basic_info(df)  # This should not raise any errors

def test_missing_values():
    # Test if missing values are printed correctly
    df = pd.read_csv('data/processed/train_data.csv')
    missing_values(df)  # This should not raise any errors

def test_plot_class_distribution():
    # Test the class distribution plot
    df = pd.read_csv('data/processed/train_data.csv')
    plot_class_distribution(df, 'label')  # This should display a plot
