# app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os

# Define the models directory path
MODELS_DIR = Path(__file__).parent / "models"

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "best_xgb_model.pkl")

model = load_model()

# Load feature names
with open(MODELS_DIR / "feature_names.txt") as f:
    model_features = [line.strip() for line in f]

st.title("🚗 Vehicle Insurance Fraud Detection")
st.markdown("Enter claim details to check for potential **fraudulent activity**.")

with st.form("fraud_form"):
    st.subheader("📝 Claim Input")

    # Numerical / Ordinal Inputs
    week_of_month = st.slider("Week of Month", 1, 5, 3)
    week_of_month_claimed = st.slider("Week of Month Claimed", 1, 5, 3)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    rep_number = st.slider("Rep Number", 1, 20, 10)
    deductible = st.selectbox("Deductible", [300, 400, 500])
    driver_rating = st.slider("Driver Rating", 1, 4, 2)
    year = st.selectbox("Year", [1994])  # Fixed since only 1994 available

    # Categorical Inputs (One-hot encoded)
    month = st.selectbox("Month", [
        "Aug", "Dec", "Feb", "Jan", "Jul", "Jun", "Mar", "May", "Nov", "Oct", "Sep"
    ])
    day_of_week = st.selectbox("Day of Week", [
        "Monday", "Saturday", "Sunday", "Thursday", "Tuesday", "Wednesday"
    ])
    make = st.selectbox("Make", [
        "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda", "Jaguar",
        "Lexus", "Mazda", "Mecedes", "Mercury", "Nisson", "Pontiac",
        "Porche", "Saab", "Saturn", "Toyota", "VW"
    ])
    accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    vehicle_category = st.selectbox("Vehicle Category", ["Sport", "Utility", "Sedan"])
    vehicle_price = st.selectbox("Vehicle Price", [
        "less than 20,000", "30,000 to 39,000", "40,000 to 59,000",
        "60,000 to 69,000", "more than 69,000"
    ])
    base_policy = st.selectbox("Base Policy", ["Collision", "Liability", "All Perils"])

    submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        # Create an input DataFrame with all expected features
        input_data = pd.DataFrame(columns=model_features)
        input_data.loc[0] = 0  # initialize all values to 0

        # Assign numeric/ordinal features
        input_data.at[0, "WeekOfMonth"] = week_of_month
        input_data.at[0, "WeekOfMonthClaimed"] = week_of_month_claimed
        input_data.at[0, "Age"] = age
        input_data.at[0, "RepNumber"] = rep_number
        input_data.at[0, "Deductible"] = deductible
        input_data.at[0, "DriverRating"] = driver_rating
        input_data.at[0, "Year"] = year

        # One-hot encoded categorical features (check existence before assignment)
        def set_one_hot(column_prefix, value):
            col_name = f"{column_prefix}_{value}"
            if col_name in input_data.columns:
                input_data.at[0, col_name] = 1

        set_one_hot("Month", month)
        set_one_hot("DayOfWeek", day_of_week)
        set_one_hot("Make", make)
        set_one_hot("AccidentArea", accident_area)
        set_one_hot("Sex", sex)
        set_one_hot("Fault", fault)
        set_one_hot("VehicleCategory", vehicle_category)
        set_one_hot("VehiclePrice", vehicle_price)
        set_one_hot("BasePolicy", base_policy)

        # Predict
        input_data = input_data[model_features].fillna(0)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.success(f"Prediction: {'🟥 Fraud' if prediction == 1 else '🟩 Non-Fraud'}")
        st.info(f"Fraud Probability: {prob:.2%}")
