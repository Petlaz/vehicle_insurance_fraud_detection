# Vehicle Insurance Fraud Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Machine learning model for detecting fraudulent vehicle insurance claims using classification algorithms and data preprocessing techniques.

## Data Source


## Project Organization

```
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

## 🚀 Getting Started

### 📦 Create and activate the environment

```bash
conda env create -f environment.yml
conda activate fraud-py311

## Quickstarts

1. **Clone the repository**

2. **Install dependencies**

pip install -r requirements.txt

3. **Train the model**

python -m python -m vehicle_insurance_fraud_detection.modeling.train

4. **Generate report**

python -m vehicle_streamlit_fraud_detection.reporting.generate_report

5. **Run prediction**

python -m vehicle_insurance_fraud_detection.modeling.predict

6. **Run Test**

pytest tests/

7. **Run Streamlit App**

streamlit run app.py

## Model Details

* **Algorithm: XGBoost**

* **Scaling: StandardScaler**

* **Test Size: 20%**


## Visualizations

Here’s the confusion matrix from the final model:


Accuracy and classification report are saved in:


## Model Metrics

**Test Accuracy:** `0.92`

**Classification Report:**

**Precision, Recall, F1-score:**  
```
📊 Classification Report (Best XGBoost):
              precision    recall  f1-score   support

   Non-Fraud       0.95      0.97      0.96      2661
       Fraud       0.34      0.19      0.24       185

    accuracy                           0.92      2846
   macro avg       0.64      0.58      0.60      2846
weighted avg       0.91      0.92      0.91      2846
```

## Deployment





## Sample Prediction

import pandas as pd

# **Replace with a real feature vector from your dataset**

sample = X_test.iloc[[0]]  # one row

# **Predict**

prediction = best_xgb.predict(sample)

probability = best_xgb.predict_proba(sample)

print("🔍 Prediction:", "Fraud" if prediction[0] == 1 else "Non-Fraud")

print("📊 Probability (Fraud):", f"{probability[0][1]:.4f}")

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SMOTE - Imbalanced-learn](https://imbalanced-learn.org/)
- [Scikit-learn Docs](https://scikit-learn.org/)
- UCI Vehicle Insurance Fraud Dataset (placeholder if used)
- Kaggle Discussions on Fraud Detection

## Requirements
- Python 3.11
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- (see `requirements.txt` for full list)


## Contributing

Pull requests are welcome. Open an issue to suggest changes or improvements.

## Contact
 
 Peter Ugonna Obi
 
 For questions, open an issue or reach out directly.

## License

MIT License.
