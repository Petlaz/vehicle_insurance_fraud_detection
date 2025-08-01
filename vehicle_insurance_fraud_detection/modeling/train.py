# vehicle_insurance_fraud_detection/modeling/train.py

import joblib
from vehicle_insurance_fraud_detection.config import MODELS_DIR

if __name__ == "__main__":
    model_path = MODELS_DIR / "best_xgb_model.pkl"
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    
# Then run this on terminal: python3 -m vehicle_insurance_fraud_detection.modeling.train
    # This script is for loading the best model after tuning and saving it.
    # It can be used for further evaluation or deployment.
