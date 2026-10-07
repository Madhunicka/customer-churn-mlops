"""
Data Ingestion, Validation, and Preprocessing Utilities.
"""
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.churn.config import DEFAULT_CONFIG, RAW_DATA_PATH, ModelConfig


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Loads raw customer churn data from a CSV file."""
    path = data_path or RAW_DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_churn_data(df: pd.DataFrame, config: ModelConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Casts TotalCharges to numeric (handles empty spaces / errors)
    - Drops nulls in TotalCharges (11 records in original Telco dataset)
    - Drops customerID if present
    - Maps target 'Churn' ('Yes' -> 1, 'No' -> 0) if present
    """
    cleaned = df.copy()

    # Convert TotalCharges to numeric
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        cleaned = cleaned.dropna(subset=["TotalCharges"]).copy()

    # Drop ID column if present
    if config.id_column in cleaned.columns:
        cleaned = cleaned.drop(columns=[config.id_column])

    # Convert target if present
    if config.target_column in cleaned.columns:
        target_series = cleaned[config.target_column]
        if target_series.dtype != int and target_series.dtype != bool:
            # Map Yes/No to 1/0
            cleaned[config.target_column] = target_series.replace({"Yes": 1, "No": 0})
        cleaned = cleaned.dropna(subset=[config.target_column]).copy()
        cleaned[config.target_column] = cleaned[config.target_column].astype(int)

    return cleaned


def split_features_and_target(
    df: pd.DataFrame, config: ModelConfig = DEFAULT_CONFIG
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates features X and target y."""
    if config.target_column not in df.columns:
        raise KeyError(f"Target column '{config.target_column}' not found in DataFrame.")

    y = df[config.target_column]
    X = df.drop(columns=[config.target_column])
    return X, y


def get_train_test_splits(
    data_path: Optional[Path] = None,
    config: ModelConfig = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads raw data, cleans it, and returns stratified train-test splits."""
    df_raw = load_raw_data(data_path)
    df_clean = clean_churn_data(df_raw, config)
    X, y = split_features_and_target(df_clean, config)

    stratify_target = y if config.stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_target,
    )

    return X_train, X_test, y_train, y_test
