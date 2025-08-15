import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# # Verify scikit-learn and joblib versions
# if sklearn.__version__ != '1.2.2':
#     st.warning(f"scikit-learn version {sklearn.__version__} detected. Version 1.2.2 is required for compatibility.")
# if joblib.__version__ != '1.2.0':
#     st.warning(f"joblib version {joblib.__version__} detected. Version 1.2.0 is required for compatibility.")

# Define the custom TargetEncoder class
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.means = None
        self.global_mean = None

    def fit(self, X, y):
        self.global_mean = y.mean()
        self.means = {}
        for col in X.columns:
            means = X.join(y).groupby(col)[y.name].mean()
            counts = X.join(y).groupby(col)[y.name].count()
            smoothed_means = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
            self.means[col] = smoothed_means
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X.columns:
            X_copy[col] = X_copy[col].map(self.means.get(col, self.global_mean)).fillna(self.global_mean)
        return X_copy

# Define file paths
PREPROCESSOR_PATH = 'D:/HUZAIFA/Car prediction/preprocessor_pipeline_final.pkl'
MODEL_PATH = 'D:/HUZAIFA/Car prediction/trained_random_forest_model1.pkl'

# Check if files exist
if not os.path.exists(PREPROCESSOR_PATH):
    st.error(f"Preprocessor file not found at {PREPROCESSOR_PATH}. Please check the path.")
    st.stop()
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    st.stop()

# Load the preprocessor and model
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# Streamlit app
st.title("Car Price Predictor")
st.write("Enter the car details to predict its price in PKR.")

# Input fields with validation
nam = st.text_input("Car Name (e.g., Toyota Corolla Altis)", value="Toyota Corolla Altis")
if not nam.strip():
    st.warning("Car Name cannot be empty.")
year = st.number_input("Year", min_value=1900, max_value=2025, value=2015, step=1)
millage = st.number_input("Mileage (km)", min_value=0.0, value=100000.0, step=1000.0)
engine_capacity = st.number_input("Engine Capacity (cc)", min_value=100.0, value=1300.0, step=100.0)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid', 'CNG', 'LPG', 'Electric'])
transmission = st.selectbox("Transmission", ['Automatic', 'Manual'])
assembly = st.selectbox("Assembly", ['Local', 'Imported'])
body_type = st.selectbox("Body Type", ['Sedan', 'Hatchback', 'SUV', 'Van', 'Truck', 'Pick Up', 'Coupe', 'Convertible', 'Micro Van', 'Mini Van', 'Double Cabin', 'Single Cabin', 'Compact sedan', 'High Roof', 'MPV', 'Station Wagon', 'Crossover', 'Compact SUV', 'Mini MPV', 'Low Roof', 'Off-Road Vehicles', 'Unknown'])
province = st.selectbox("Province", ['Punjab', 'Sindh', 'KPK', 'Balochistan', 'Islamabad Capital Territory'])
color = st.selectbox("Color", ['White', 'Silver', 'Black', 'Grey', 'Blue', 'Red', 'Green', 'Gold', 'Beige', 'Brown', 'Maroon', 'Unknown'])
features = st.multiselect("Features", ['Power Steering', 'Power Windows', 'Air Conditioning', 'ABS', 'Air Bags', 'Power Locks', 'Power Mirrors', 'Alloy Rims', 'Keyless Entry', 'Immobilizer Key'])

# Create input DataFrame
input_data = pd.DataFrame({
    'nam': [nam.strip()],
    'Year': [year],
    'Millage': [millage],
    'Engine Capacity': [engine_capacity],
    'Fuel': [fuel],
    'Transmission': [transmission],
    'Assembly': [assembly],
    'Body Type': [body_type],
    'Province': [province],
    'Color': [color]
})

# Add binary feature columns
for feature in ['Power Steering', 'Power Windows', 'Air Conditioning', 'ABS', 'Air Bags', 'Power Locks', 'Power Mirrors', 'Alloy Rims', 'Keyless Entry', 'Immobilizer Key']:
    input_data[f'has_{feature.replace(" ", "_").replace("/", "_")}'] = [1 if feature in features else 0]

# Predict button
if st.button("Predict Price"):
    if not nam.strip():
        st.error("Please enter a valid Car Name.")
    else:
        try:
            # Transform input data
            X_transformed = preprocessor.transform(input_data)
            # Predict
            price_pred = model.predict(X_transformed)[0]
            # Ensure positive price
            price_pred = max(0, price_pred)
            # Display result
            st.success(f"Predicted Car Price: {price_pred:,.2f} PKR")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Instructions
st.write("""
**Instructions:**
1. Enter the car details above (ensure Car Name is not empty).
2. Select all applicable features.
3. Click 'Predict Price' to see the estimated price in PKR.
""")