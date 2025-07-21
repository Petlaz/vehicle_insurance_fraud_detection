🚗 Vehicle Insurance Fraud Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A machine learning project for detecting fraudulent vehicle insurance claims using classification algorithms and modern data preprocessing techniques.


📂 Data Source
Dataset: Kaggle – Vehicle Insurance Fraud Detection Dataset


🧭 Project Structure

```
Vehicle_insurance_fraud_detection
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         vehicle_insurance_fraud_detection and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── vehicle_insurance_fraud_detection   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes vehicle_insurance_fraud_detection a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

🚀 Getting Started

1. Create & activate environment

```bash
conda env create -f environment.yml

conda activate fraud-py311

2. Clone and install dependencies

git clone https://github.com/petlaz/vehicle_insurance_fraud_detection.git
cd vehicle_insurance_fraud_detection
pip install -r requirements.txt


 ⚙️ Quickstart Commands

| Task              | Command                                                                 |
| ----------------- | ----------------------------------------------------------------------- |
| Train the model   | `python -m vehicle_insurance_fraud_detection.modeling.train`            |
| Run prediction    | `python -m vehicle_insurance_fraud_detection.modeling.predict`          |
| Generate reports  | `python -m vehicle_insurance_fraud_detection.reporting.generate_report` |
| Run tests         | `pytest tests/`                                                         |
| Run Streamlit app | `streamlit run app.py`                                                  |

🧠 Model Details

- Algorithm: XGBoost

- Scaling: StandardScaler

- Sampling Strategy: SMOTE

- Evaluation Metric: F1-score

- Test Size: 20%


📊 Model Evaluation

Classification Reports

classification_report.txt

classification_report.json


📈 Visual Reports

- ![Confusion Matrix](reports/figures/confusion_matrix.png)

- ![ROC Curve](reports/figures/roc_curve.png)

- ![Feature Importance](reports/figures/feature_importance.png)


🌐 Deployment

* The app is deployed with Streamlit

* Launch:

streamlit run app.py


🔮 Sample Prediction

import pandas as pd

# Replace with actual feature vector

sample = X_test.iloc[[0]]

# Predict

prediction = best_xgb.predict(sample)

probability = best_xgb.predict_proba(sample)

print("🔍 Prediction:", "Fraud" if prediction[0] == 1 else "Non-Fraud")

print("📊 Probability (Fraud):", f"{probability[0][1]:.4f}")


🔗 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)

- [SMOTE - Imbalanced-learn](https://imbalanced-learn.org/)

- [Scikit-learn Docs](https://scikit-learn.org/)

- Kaggle Discussions on Fraud Detection


📦 Requirements

- Python 3.11
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- (see `requirements.txt` for full list)


🤝 Contributing

Pull requests are welcome. Open an issue to suggest changes or improvements.


📬 Contact

Peter Ugonna Obi
For questions or feedback, open an issue or reach out directly.


📄 License

This project is licensed under the MIT License.


