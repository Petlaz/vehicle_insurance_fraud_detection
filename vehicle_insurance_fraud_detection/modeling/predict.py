# vehicle_insurance_fraud_detection/modeling/predict.py

import joblib
import pandas as pd
import numpy as np
from vehicle_insurance_fraud_detection.config import MODELS_DIR
from vehicle_insurance_fraud_detection.dataset import load_clean_data
from sklearn.metrics import classification_report

def predict_on_test_sample():
    # Load model
    model_path = MODELS_DIR / "best_xgb_model.pkl"
    model = joblib.load(model_path)

    # Load cleaned data
    df = load_clean_data()

    # Split into features and target
    X = df.drop("FraudFound", axis=1)
    y = df["FraudFound"]

    # Pick a test sample
    sample = X.iloc[[0]]  # Example: First row

    # Predict
    prediction = model.predict(sample)
    probability = model.predict_proba(sample)[0][1]

    print("🔍 Prediction:", "Fraud" if prediction[0] == 1 else "Non-Fraud")
    print("📊 Probability (Fraud):", f"{probability:.4f}")

if __name__ == "__main__":
    predict_on_test_sample()
    
    # Then run this on terminal: python3 -m vehicle_insurance_fraud_detection.modeling.predict
    # This script is for making predictions on a single test sample using the best model.
    # It loads the model, the cleaned data, and prints the prediction and probability.