"""Model training pipeline."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from data_loader import load_data, save_data
from preprocessing import clean_data, add_features


def build_pipeline(model, cat_cols, num_cols):
    """Build sklearn pipeline with preprocessing + model."""
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    return Pipeline([('prep', preprocessor), ('model', model)])


def evaluate(y_true, y_pred, name):
    """Print evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {name}")
    print(f"   RMSE: {rmse:,.0f}")
    print(f"   MAE : {mae:,.0f}")
    print(f"   R²  : {r2:.4f}")
    return {'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    # 1. Load & clean
    df = load_data('data/raw/pakistan_used_cars.csv')
    df = clean_data(df)
    df = add_features(df)
    save_data(df, 'data/processed/cleaned_cars.csv')

    # 2. Define features
    target = 'price'
    drop_cols = ['price', 'log_price', 'price_per_cc']
    features = [c for c in df.columns if c not in drop_cols]

    cat_cols = df[features].select_dtypes(include='object').columns.tolist()
    num_cols = df[features].select_dtypes(exclude='object').columns.tolist()

    X = df[features]
    y = df[target]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train models
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            random_state=42, n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=10,
            num_leaves=64, random_state=42, n_jobs=-1
        ),
    }

    results = []
    best_model, best_score = None, -np.inf

    for name, model in models.items():
        pipe = build_pipeline(model, cat_cols, num_cols)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        res = evaluate(y_test, preds, name)
        results.append(res)
        if res['r2'] > best_score:
            best_score = res['r2']
            best_model = pipe
            best_name = name

    # 5. Save best model
    Path('models').mkdir(exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\n🏆 Best model: {best_name} (R² = {best_score:.4f})")
    print("💾 Saved to models/best_model.pkl")

    # 6. Save results
    pd.DataFrame(results).to_csv('reports/model_comparison.csv', index=False)


if __name__ == '__main__':
    main()