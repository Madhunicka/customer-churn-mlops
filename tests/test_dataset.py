"""
Unit Tests for Data Ingestion, Cleaning, and Preprocessing.
"""
import pandas as pd
import pytest

from src.churn.config import DEFAULT_CONFIG
from src.churn.dataset import clean_churn_data, get_train_test_splits, split_features_and_target


@pytest.fixture
def sample_raw_df():
    """Provides a realistic sample raw DataFrame resembling Telco Churn."""
    return pd.DataFrame({
        "customerID": ["001-AAA", "002-BBB", "003-CCC", "004-DDD"],
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 34, 2, 45],
        "PhoneService": ["No", "Yes", "Yes", "No"],
        "MultipleLines": ["No phone service", "No", "No", "No phone service"],
        "InternetService": ["DSL", "DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No", "Yes"],
        "OnlineBackup": ["Yes", "No", "No", "No"],
        "DeviceProtection": ["No", "Yes", "No", "Yes"],
        "TechSupport": ["No", "No", "No", "Yes"],
        "StreamingTV": ["No", "No", "No", "No"],
        "StreamingMovies": ["No", "No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Electronic check",
            "Bank transfer (automatic)",
        ],
        "MonthlyCharges": [29.85, 56.95, 70.70, 42.30],
        "TotalCharges": ["29.85", "1889.5", " ", "1840.75"],  # contains blank space
        "Churn": ["No", "No", "Yes", "No"],
    })


def test_clean_churn_data(sample_raw_df):
    """Verifies that clean_churn_data drops blanks, drops customerID, and maps Churn."""
    cleaned = clean_churn_data(sample_raw_df, DEFAULT_CONFIG)

    # 1 row had blank TotalCharges -> 3 rows remain
    assert len(cleaned) == 3
    assert DEFAULT_CONFIG.id_column not in cleaned.columns
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_split_features_and_target(sample_raw_df):
    """Verifies proper separation of X and y."""
    cleaned = clean_churn_data(sample_raw_df, DEFAULT_CONFIG)
    X, y = split_features_and_target(cleaned, DEFAULT_CONFIG)

    assert DEFAULT_CONFIG.target_column not in X.columns
    assert len(X) == len(y)
    assert len(X) == 3


def test_get_train_test_splits():
    """Integration test verifying train/test split generation on actual raw CSV."""
    X_train, X_test, y_train, y_test = get_train_test_splits()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) > 7000
    assert X_train.shape[1] == 19  # 19 features (excluding customerID and Churn)
    assert set(y_train.unique()) == {0, 1}
