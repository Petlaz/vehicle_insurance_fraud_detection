# tests/test_data.py

import pytest
import pandas as pd
from vehicle_insurance_fraud_detection.dataset import load_clean_data
from vehicle_insurance_fraud_detection.features import split_features_targets

def test_load_clean_data():
    df = load_clean_data()
    
    # Check that it's a DataFrame
    assert isinstance(df, pd.DataFrame), "Loaded data should be a DataFrame"

    # Check for non-zero rows and columns
    assert df.shape[0] > 0, "DataFrame should have at least one row"
    assert df.shape[1] > 1, "DataFrame should have at least two columns"

    # Check that target column exists
    assert 'FraudFound' in df.columns, "'FraudFound' target column must exist"

    # Check for missing values
    assert df.isnull().sum().sum() == 0, "DataFrame should not contain missing values"

def test_split_features_targets():
    df = load_clean_data()
    X, y = split_features_targets(df)

    # X and y should have the same number of rows
    assert len(X) == len(y), "Features and target should have the same number of samples"

    # y should be 1-dimensional
    assert y.ndim == 1, "Target should be 1-dimensional"

    # Check that the target column is not in X
    assert 'FraudFound' not in X.columns, "'FraudFound' should not be in features"

# Run the tests: pytest tests/test_data.py
