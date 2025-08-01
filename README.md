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

vehicle_insurance_fraud_detection/

- data/
  - raw/              # Original immutable data
  - processed/        # Cleaned data for modeling
  - interim/          # Intermediate transformations

- models/             # Serialized models

- notebooks/          # Jupyter notebooks (e.g., 1.0-eda.ipynb)

- reports/            # Analysis outputs
  - figures/          # Visualizations

- src/                # Python package
  - __init__.py
  - config.py         # Project configurations
  - features.py       # Feature engineering
  - modeling/         # ML pipelines
    - train.py
    - predict.py

- app.py              # Streamlit application

- environment.yml     # Conda environment

- requirements.txt    # Pip dependencies

## 🚀 Quick Start  

**Installation**:  
```bash
git clone https://github.com/petlaz/vehicle_insurance_fraud_detection.git
cd vehicle_insurance_fraud_detection
conda env create -f environment.yml
conda activate fraud-py311
pip install -r requirements.txt

Commands

| Action           | Command                          |
| ---------------- | -------------------------------- |
| Preprocess data  | `make data`                      |
| Train model      | `python -m src.modeling.train`   |
| Make predictions | `python -m src.modeling.predict` |
| Launch dashboard | `streamlit run app.py`           |


# 📈 Model Performance

### 🔧 Configuration

- **Algorithm**: `XGBoostClassifier`  
- **Hyperparameter Tuning**: `Optuna`  
- **Class Balancing**: `SMOTE`  
- **Test Size**: `20%`

### 📊 Metrics

| Metric    | Score |
|-----------|-------|
| Accuracy  | 92%   |
| Precision | 0.91  |
| Recall    | 0.87  |
| F1-Score  | 0.89  |
| AUC-ROC   | 0.93  |


# 🌐 Deployment

## Cloud Deployment

The vehicle insurance fraud detection app is live and hosted on **Streamlit Cloud**:

🔗 [Live App on Streamlit Cloud](https://vehicleinsurancefrauddetection-mmqyvhriq3jdmluu3tpqog.streamlit.app/)

---

## Local Deployment

To run the app locally on your machine, follow these steps:

1. Ensure you have all dependencies installed (see `requirements.txt` or `environment.yml`).

2. From the project root directory, run:

```bash
streamlit run app.py


# 🔮 Usage Example

This example demonstrates how to load the trained XGBoost model and make a prediction on a sample input.

## Prerequisites

- Python 3.11
- Required packages installed (`joblib`, `pandas`, etc.)
- Trained model saved as `models/xgboost_model.pkl`

---

## Example Code

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/xgboost_model.pkl")

# Prepare a sample input DataFrame
sample = pd.DataFrame({
    'claim_amount': [5000],
    'vehicle_age': [3],
    'past_claims': [2]
})

# Make prediction
prediction = model.predict(sample)

# Get fraud probability
probability = model.predict_proba(sample)[0][1]

print(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
print(f"Fraud Probability: {probability:.1%}")


# 🛣️ Roadmap

This roadmap outlines the planned features and improvements for the Vehicle Insurance Fraud Detection project.

## Upcoming Features

- **Initial Model Pipeline**  
  Establish a robust and reproducible machine learning pipeline for data preprocessing, model training, and evaluation.

- **SHAP / LIME Explainability**  
  Integrate explainability tools like SHAP and LIME to provide insights into model predictions and feature importance.

- **FastAPI Backend**  
  Develop a FastAPI backend service to serve the model via REST API, enabling scalable and flexible integration.

- **User Authentication**  
  Implement user authentication and authorization for the application to secure access and enable personalized features.

## Future Enhancements

- Real-time fraud detection with streaming data support  
- Advanced ensemble modeling and hyperparameter tuning automation  
- Dashboard enhancements with additional visualization and interactivity  
- Integration with external fraud databases and alerts  

---

Contributions and suggestions are welcome to help evolve this roadmap!


# 👤 Peter Ugonna Obi

Peter Ugonna Obi is the author and maintainer of this Vehicle Insurance Fraud Detection project.

## Contact Information

- **Email:** peter.obi96@yahoo.com  
- **LinkedIn:** [linkedin.com/in/peter-obi-15a424161](https://linkedin.com/in/peter-obi-15a424161)

## About Me

I am passionate about machine learning, data science, and building impactful AI solutions. This project reflects my commitment to developing practical and explainable models for real-world problems such as insurance fraud detection.

Feel free to reach out with questions, suggestions, or collaboration opportunities!

---

Thank you for your interest and support.


# 📜 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for more details.

---

You are free to use, modify, and distribute this software under the terms of the MIT License.


