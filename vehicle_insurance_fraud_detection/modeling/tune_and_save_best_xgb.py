# vehicle_insurance_fraud_detection/modeling/tune_and_save_best_xgb.py

import time
import joblib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from vehicle_insurance_fraud_detection.dataset import load_clean_data
from vehicle_insurance_fraud_detection.features import split_features_targets
from vehicle_insurance_fraud_detection.config import MODELS_DIR


class XGBoostTuner:
    def __init__(self, random_state=42):
        self.best_model = None
        self.random_state = random_state

    def tune(self, X, y, n_iter=10, cv=3, n_jobs=2, verbose=1):
        print(f"\n🚀 Tuning XGBoost on {X.shape[0]} samples, {X.shape[1]} features...")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=1,
                tree_method='hist',
                random_state=self.random_state
            ))
        ])

        param_dist = {
            'xgb__n_estimators': randint(100, 200),
            'xgb__max_depth': randint(3, 10),
            'xgb__learning_rate': uniform(0.01, 0.2),
            'xgb__subsample': uniform(0.6, 0.4),
            'xgb__colsample_bytree': uniform(0.6, 0.4),
            'xgb__gamma': uniform(0, 0.5),
            'xgb__reg_alpha': uniform(0, 1),
            'xgb__reg_lambda': uniform(0, 1),
            'xgb__min_child_weight': randint(1, 6)
        }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1',
            cv=cv,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )

        start = time.time()
        search.fit(X, y)
        duration = time.time() - start

        print(f"\n✅ Best F1 Score: {search.best_score_:.4f}")
        print("🔧 Best Parameters:")
        print(search.best_params_)
        print(f"⏱️ Duration: {duration/60:.2f} minutes")

        # Save model
        self.best_model = search.best_estimator_
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, MODELS_DIR / "best_xgb_model.pkl")
        print(f"📦 Saved to {MODELS_DIR / 'best_xgb_model.pkl'}")

        # Save feature names
        feature_names = X.columns.tolist()
        with open(MODELS_DIR / "feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"📝 Feature names saved to {MODELS_DIR / 'feature_names.txt'}")

        return self.best_model


if __name__ == "__main__":
    df = load_clean_data()
    X, y = split_features_targets(df)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    tuner = XGBoostTuner()
    best_model = tuner.tune(X_smote, y_smote)

    print("🏆 Best model tuned and saved successfully.")

    # Run via: python -m vehicle_insurance_fraud_detection.modeling.tune_and_save_best_xgb
    # This script tunes the XGBoost model using RandomizedSearchCV and saves the best model.
    # It also saves the feature names used during training for later use in predictions.
    # Ensure you have the necessary libraries installed: xgboost, imbalanced-learn, scikit-learn, pandas, matplotlib, seaborn.
    # The model is saved in the `models` directory as `best_xgb_model.pkl`.
    # Feature names are saved in `models/feature_names.txt`.
    # The script also prints the best parameters and F1 score achieved during tuning.
    # It uses SMOTE to handle class imbalance in the dataset before tuning.
    # The tuning process includes hyperparameter optimization for the XGBoost classifier.
    # The script is designed to be run as a module, ensuring proper execution context.
    # It prints the duration of the tuning process and saves the tuned model and feature names for later use.
    # The model can be loaded using joblib for predictions in a Streamlit app or other applications.
    # The script is structured to be reusable and modular, allowing for easy integration
    # into larger projects or pipelines.