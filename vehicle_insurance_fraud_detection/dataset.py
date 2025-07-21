# vehicle_insurance_fraud_detection/dataset.py

import os
import pandas as pd
from vehicle_insurance_fraud_detection.config import PROJ_ROOT

RAW_DATA_PATH = "data/raw/vehicle_insurance_fraud_detection.csv"
PROCESSED_PATH = "data/processed/data_df_encoded.csv"

def prepare_data_folders(base_path="data"):
    """Ensure the raw/interim/processed/external folders exist."""
    folders = ["raw", "interim", "processed", "external"]
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
    print("✅ Data folders created or already exist.")

def load_dataset(path=RAW_DATA_PATH):
    """Load the raw dataset."""
    return pd.read_csv(path)

def load_clean_data(path=PROCESSED_PATH):
    """Load the cleaned and encoded dataset from processed folder."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned dataset not found at {path}")
    return pd.read_csv(path)

def split_features_targets(df):
    """Split dataframe into X (features) and y (target)."""
    X = df.drop("FraudFound", axis=1)
    y = df["FraudFound"]
    return X, y

if __name__ == "__main__":
    prepare_data_folders()
    df = load_dataset()
    print("📄 Dataset preview:")
    print(df.head())
