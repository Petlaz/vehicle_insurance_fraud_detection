# tests/test_model.py

import pytest
import joblib
import pandas as pd
from vehicle_insurance_fraud_detection.config import MODELS_DIR
from vehicle_insurance_fraud_detection.dataset import load_clean_data
from vehicle_insurance_fraud_detection.features import split_features_targets

# Load feature names used during training
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.txt"

@pytest.fixture(scope="module")
def model():
    model_path = MODELS_DIR / "best_xgb_model.pkl"
    loaded_model = joblib.load(model_path)
    return loaded_model

@pytest.fixture(scope="module")
def sample_data():
    df = load_clean_data()
    X, y = split_features_targets(df)
    return X.iloc[:5]  # use only a few rows for speed

def test_model_load(model):
    assert model is not None, "Model should load successfully"

def test_model_predict_shape(model, sample_data):
    preds = model.predict(sample_data)
    assert preds.shape[0] == sample_data.shape[0], "Prediction shape mismatch"

def test_model_predict_valid_values(model, sample_data):
    preds = model.predict(sample_data)
    assert set(preds).issubset({0, 1}), "Predictions should only be 0 or 1"

def test_model_feature_alignment(sample_data):
    with open(FEATURE_NAMES_PATH) as f:
        feature_names = [line.strip() for line in f]
    
    missing = set(feature_names) - set(sample_data.columns)
    extra = set(sample_data.columns) - set(feature_names)

    assert not missing, f"Missing expected features: {missing}"
    assert not extra, f"Unexpected extra features: {extra}"

# Run the tests: pytest tests/test_model.py