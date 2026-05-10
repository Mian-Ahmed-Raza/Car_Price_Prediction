"""Streamlit deployment app."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🚗 Car Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

model = load_model()

st.title("🚗 Pakistani Used Car Price Predictor")
st.markdown("Estimate the fair market price of a used car in Pakistan")

col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox("Make", ['Toyota', 'Honda', 'Suzuki', 'Hyundai', 'Kia', 'Nissan'])
    model_name = st.text_input("Model", "Corolla")
    year = st.slider("Year", 1990, 2022, 2018)
    engine = st.number_input("Engine (cc)", 600, 6000, 1300, step=100)

with col2:
    city = st.selectbox("City", ['Lahore', 'Karachi', 'Islamabad', 'Peshawar', 'Rawalpindi'])
    body = st.selectbox("Body", ['Sedan', 'Hatchback', 'SUV', 'Crossover', 'Van'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    fuel = st.selectbox("Fuel", ['Petrol', 'Diesel', 'Hybrid', 'Cng', 'Lpg'])

with col3:
    color = st.selectbox("Color", ['White', 'Black', 'Silver', 'Grey', 'Blue', 'Red'])
    assembly = st.selectbox("Assembly", ['Local', 'Imported', 'Unknown'])
    registered = st.selectbox("Registered", ['Lahore', 'Karachi', 'Islamabad', 'Punjab', 'Sindh'])
    mileage = st.number_input("Mileage (km)", 0, 500000, 70000, step=5000)

if st.button("🔮 Predict Price", type="primary"):
    input_data = {
        'city': city, 'assembly': assembly, 'body': body, 'make': make,
        'model': model_name, 'year': year, 'engine': engine,
        'transmission': transmission, 'fuel': fuel, 'color': color,
        'registered': registered, 'mileage': mileage,
        'car_age': 2022 - year,
        'mileage_per_year': mileage / (2022 - year + 1),
        'is_imported': 1 if assembly == 'Imported' else 0,
        'is_automatic': 1 if transmission == 'Automatic' else 0,
        'log_mileage': np.log1p(mileage)
    }
    df = pd.DataFrame([input_data])
    price = model.predict(df)[0]

    st.success(f"### 💰 Estimated Price: **PKR {price:,.0f}**")
    st.info(f"≈ USD ${price/280:,.0f}")