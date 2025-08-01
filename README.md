# 🚗 Vehicle Insurance Fraud Detection  

[![Cookiecutter Data Science](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://vehicleinsurancefrauddetection-mmqyvhriq3jdmluu3tpqog.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Overview 

Machine learning system to detect fraudulent vehicle insurance claims using:

- **XGBoost** for classification  

- **SMOTE** for handling class imbalance 

- **MLOps** best practices for reproducibility

## 📂 Dataset  

**Source**: [Kaggle Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/)

**Features**: 

- Claim details

- Policyholder information

- Vehicle specifications 

- Historical transaction data  

**Target Variable**:

`fraud_label` (0 = Legitimate, 1 = Fraud)


## 🏗️ Project Structure 

,,,
vehicle_insurance_fraud_detection/
├── data/
│ ├── raw/ # Original immutable data
│ ├── processed/ # Cleaned data for modeling
│ └── interim/ # Intermediate transformations
├── models/ # Serialized models
├── notebooks/ # Jupyter notebooks (1.0-eda.ipynb)
├── reports/ # Analysis outputs
│ └── figures/ # Visualizations
├── src/ # Python package
│ ├── init.py
│ ├── config.py # Project configurations
│ ├── features.py # Feature engineering
│ └── modeling/ # ML pipelines
│ ├── train.py
│ └── predict.py
├── app.py # Streamlit application
├── environment.yml # Conda environment
└── requirements.txt # Pip dependencies
,,,

## 🚀 Quick Start  

**Installation**:  
```bash

git clone https://github.com/petlaz/vehicle_insurance_fraud_detection.git

cd vehicle_insurance_fraud_detection

conda env create -f environment.yml

conda activate fraud-py311

pip install -r requirements.txt

Commands

Action	Command

Preprocess data	make data

Train model	python -m src.modeling.train

Make predictions	python -m src.modeling.predict

Launch dashboard	streamlit run app.py

## 📈 Model Performance

### 🔧 Configuration
- **Algorithm**: `XGBoostClassifier`
- **Hyperparameter Tuning**: `Optuna`
- **Class Balancing**: `SMOTE`
- **Test Size**: `20%`

### 📊 Metrics

| Metric     | Score |
|------------|-------|
| Accuracy   | 92%   |
| Precision  | 0.91  |
| Recall     | 0.87  |
| F1-Score   | 0.89  |
| AUC-ROC    | 0.93  |

---


## 🌐 Deployment

- **Cloud App**: [Live on Streamlit Cloud](#) <!https://vehicleinsurancefrauddetection-mmqyvhriq3jdmluu3tpqog.streamlit.app/ -->

- **Run Locally**:

```bash
streamlit run app.py

🔮 Usage Example

python

import joblib

import pandas as pd

# Load model

model = joblib.load("models/xgboost_model.pkl")

# Sample prediction

sample = pd.DataFrame({
    'claim_amount': [5000],
    'vehicle_age': [3],
    'past_claims': [2]
})

pred = model.predict(sample)

prob = model.predict_proba(sample)[0][1]

print(f"Prediction: {'Fraud' if pred[0] == 1 else 'Legitimate'}")

print(f"Fraud Probability: {prob:.1%}")

## 📌 Roadmap

* Initial model pipeline

* SHAP/LIME explainability

* FastAPI backend

* User authentication

## 🤝 Contributing

- Fork the repository

- Create your feature branch

Submit a pull request

📬 Contact

## Peter Ugonna Obi

* Email: peter.obi96@yahoo.com

* LinkedIn: linkedin.com/in/peter-obi-15a424161

## 📜 License

MIT License. See LICENSE for details.

