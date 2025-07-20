# reports/generate_figures.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from vehicle_insurance_fraud_detection.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

from vehicle_insurance_fraud_detection.dataset import load_clean_data
from vehicle_insurance_fraud_detection.features import split_features_targets
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load model
model = joblib.load("best_xgb_model.pkl")

# Load and prepare data
df = load_clean_data()
X, y = split_features_targets(df)
X, y = SMOTE(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig_cm = plot_confusion_matrix(cm)
fig_cm.write_image("reports/figures/confusion_matrix.png")

# 2. ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig_roc = plot_roc_curve(fpr, tpr, roc_auc)
fig_roc.write_image("reports/figures/roc_curve.png")

# 3. Feature Importance
importances = model.named_steps['xgb'].feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
fig_imp = plot_feature_importance(importance_df)
fig_imp.write_image("reports/figures/feature_importance.png")

print("✅ All figures saved to reports/figures/")
