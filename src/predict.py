"""Inference utility."""
import pandas as pd
import joblib

def predict_price(input_dict: dict, model_path='models/best_model.pkl') -> float:
    """Predict price for a single car."""
    model = joblib.load(model_path)
    df = pd.DataFrame([input_dict])

    # Add engineered features (must match training)
    df['car_age'] = 2022 - df['year']
    df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
    df['is_imported'] = (df['assembly'].str.lower() == 'imported').astype(int)
    df['is_automatic'] = (df['transmission'] == 'Automatic').astype(int)
    import numpy as np
    df['log_mileage'] = np.log1p(df['mileage'])

    return float(model.predict(df)[0])


if __name__ == '__main__':
    sample = {
        'city': 'Lahore', 'assembly': 'Local', 'body': 'Sedan',
        'make': 'Toyota', 'model': 'Corolla', 'year': 2018,
        'engine': 1300, 'transmission': 'Automatic', 'fuel': 'Petrol',
        'color': 'White', 'registered': 'Lahore', 'mileage': 70000
    }
    price = predict_price(sample)
    print(f"💰 Predicted Price: PKR {price:,.0f}")