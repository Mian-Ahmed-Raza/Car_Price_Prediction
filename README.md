# 🚗 Pakistani Used Car Price Prediction

## 📌 Project Overview
A machine learning project that predicts the resale price of used cars in Pakistan
based on features like make, model, year, engine size, mileage, fuel type, and location.

## 🎯 Objectives
- Perform deep EDA on Pakistani car market trends
- Build a robust preprocessing pipeline
- Train and compare multiple ML models
- Deploy via a Streamlit web app

## 📊 Dataset
- **Source**: PakWheels.com listings
- **Records**: 77,878
- **Features**: 14
- **Target**: `price` (PKR)

## 🛠️ Tech Stack
- Python 3.9+
- pandas, numpy, scikit-learn
- XGBoost, LightGBM, CatBoost
- matplotlib, seaborn, plotly
- Streamlit (deployment)
- joblib (model persistence)

## 🚀 How to Run
```bash
# Clone repo
git clone https://github.com/yourusername/used-car-price-prediction.git
cd used-car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train_model.py

# Run web app
streamlit run app.py
