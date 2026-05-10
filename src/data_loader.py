"""Data loading utilities."""
import pandas as pd
from pathlib import Path

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def save_data(df: pd.DataFrame, filepath: str):
    """Save processed data."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"💾 Saved to {filepath}")