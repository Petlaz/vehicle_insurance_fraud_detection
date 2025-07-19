# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from vehicle_insurance_fraud_detection.config import MODELS_DIR

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "best_xgb_model.pkl")

model = load_model()

# Load feature names exactly as used during training
with open(MODELS_DIR / "feature_names.txt") as f:
    model_features = [line.strip() for line in f]

st.title("🚗 Vehicle Insurance Fraud Detection")
st.markdown("Enter claim details to check for potential **fraudulent activity**.")

with st.form("fraud_form"):
    st.subheader("📝 Claim Input")

    week_of_month = st.slider("Week of Month", 1, 5, 3)
    week_of_month_claimed = st.slider("Week of Month Claimed", 1, 5, 3)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    rep_number = st.slider("Rep Number", 1, 20, 10)
    deductible = st.selectbox("Deductible", [300, 400, 500])
    driver_rating = st.slider("Driver Rating", 1, 4, 2)
    year = st.selectbox("Year", [1994])

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
        input_data = pd.DataFrame(columns=model_features)
        input_data.loc[0] = 0

        input_data["WeekOfMonth"] = week_of_month
        input_data["WeekOfMonthClaimed"] = week_of_month_claimed
        input_data["Age"] = age
        input_data["RepNumber"] = rep_number
        input_data["Deductible"] = deductible
        input_data["DriverRating"] = driver_rating
        input_data["Year"] = year

        input_data[f"Month_{month}"] = 1
        input_data[f"DayOfWeek_{day_of_week}"] = 1
        input_data[f"Make_{make}"] = 1
        input_data[f"AccidentArea_{accident_area}"] = 1
        input_data[f"Sex_{sex}"] = 1
        input_data[f"Fault_{fault}"] = 1
        input_data[f"VehicleCategory_{vehicle_category}"] = 1
        input_data[f"VehiclePrice_{vehicle_price}"] = 1
        input_data[f"BasePolicy_{base_policy}"] = 1

        input_data.fillna(0, inplace=True)
        input_data = input_data[model_features]

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.success(f"Prediction: {'🟥 Fraud' if prediction == 1 else '🟩 Non-Fraud'}")
        st.info(f"Fraud Probability: {prob:.2%}")

# Run the Streamlit app with: streamlit run streamlit_app.py
# This app allows users to input claim details and get predictions on potential fraud.
# It uses the best XGBoost model trained and saved in the modeling directory.
# The app caches the model loading for performance and uses a form for user input.
# Ensure you have the necessary libraries installed: streamlit, pandas, joblib, xgboost.
# The app provides a user-friendly interface to input claim details and receive predictions.
# The model features are dynamically loaded from a text file to ensure consistency with training.
# The app is designed to be run in a Streamlit environment, providing an interactive experience.
# The input form includes sliders, select boxes, and number inputs for various claim attributes.
# The prediction results are displayed with appropriate styling for clarity.
# The app can be extended with more features or visualizations as needed.
