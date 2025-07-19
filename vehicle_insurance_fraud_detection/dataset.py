# vehicle_insurance_fraud_detection/dataset.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
from vehicle_insurance_fraud_detection.config import PROJ_ROOT


RAW_DATA_PATH = "data/raw/vehicle_insurance_fraud_detection.csv"


def prepare_data_folders(base_path="data"):
    folders = ["raw", "interim", "processed", "external"]
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
    print("✅ Data folders created or already exist.")


def load_dataset():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


if __name__ == "__main__":
    prepare_data_folders()
    df = load_dataset()
    print("📄 Dataset preview:")
    print(df.head())

PROCESSED_PATH = "data/processed/data_df_encoded.csv"

def load_clean_data():
    """Load the cleaned and encoded dataset from processed folder."""
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH)
    return df

    
