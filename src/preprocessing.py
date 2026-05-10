"""Data cleaning and preprocessing pipeline."""
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive data cleaning."""
    df = df.copy()

    # 1. Drop ID column
    if 'addref' in df.columns:
        df.drop(columns=['addref'], inplace=True)

    # 2. Drop rows with missing target
    df = df.dropna(subset=['price'])

    # 3. Handle 'assembly' (~69% missing) — fill with 'Unknown'
    df['assembly'] = df['assembly'].fillna('Unknown')

    # 4. Fill categorical missing values with mode
    for col in ['body', 'fuel', 'color']:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Fill 'year' with median by make/model
    df['year'] = df.groupby(['make', 'model'])['year'].transform(
        lambda x: x.fillna(x.median())
    )
    df['year'] = df['year'].fillna(df['year'].median())

    # 6. Outlier removal
    df = df[(df['engine'] >= 600) & (df['engine'] <= 8000)]
    df = df[(df['mileage'] >= 100) & (df['mileage'] <= 500000)]
    df = df[(df['price'] >= 100000) & (df['price'] <= 50000000)]
    df = df[df['year'] >= 1990]

    # 7. Remove placeholder mileage values
    df = df[df['mileage'] != 123456]

    # 8. Standardize text columns
    text_cols = ['city', 'make', 'model', 'body', 'transmission', 'fuel', 'registered']
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.title()

    print(f"✅ After cleaning: {df.shape[0]} rows remain")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    df = df.copy()
    current_year = 2022

    df['car_age'] = current_year - df['year']
    df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
    df['price_per_cc'] = df['price'] / df['engine']
    df['is_imported'] = (df['assembly'].str.lower() == 'imported').astype(int)
    df['is_automatic'] = (df['transmission'] == 'Automatic').astype(int)

    # Log-transform skewed numeric features
    df['log_mileage'] = np.log1p(df['mileage'])
    df['log_price'] = np.log1p(df['price'])

    # Group rare models (< 50 occurrences) as 'Other'
    model_counts = df['model'].value_counts()
    rare_models = model_counts[model_counts < 50].index
    df['model'] = df['model'].replace(rare_models, 'Other')

    print(f"✅ Engineered features added. Shape: {df.shape}")
    return df