# reports/generate_metrics.py

import joblib
import json
from sklearn.metrics import classification_report
from vehicle_insurance_fraud_detection.dataset import load_clean_data
from vehicle_insurance_fraud_detection.features import split_features_targets
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load model and data
model = joblib.load("best_xgb_model.pkl")
df = load_clean_data()
X, y = split_features_targets(df)
X, y = SMOTE(random_state=42).fit_resample(X, y)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict and generate report
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Save to .json
with open("reports/metrics/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

# Save to .txt
from sklearn.metrics import classification_report
text_report = classification_report(y_test, y_pred)
with open("reports/metrics/classification_report.txt", "w") as f:
    f.write(text_report)

print("✅ Classification report saved in JSON and TXT formats.")
