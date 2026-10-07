"""
Centralized Configuration for the Churn Prediction MLOps Pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


@dataclass
class ModelConfig:
    # Feature Definitions
    id_column: str = "customerID"
    target_column: str = "Churn"

    numerical_features: List[str] = field(
        default_factory=lambda: [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]
    )

    categorical_features: List[str] = field(
        default_factory=lambda: [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
    )

    # Training Hyperparameters
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # Logistic Regression Defaults
    lr_max_iter: int = 1000
    lr_c: float = 1.0

    # Random Forest Defaults
    rf_n_estimators: int = 200
    rf_max_depth: int = 10

    # Model Artifact Path
    model_artifact_path: Path = MODELS_DIR / "customer_churn_model.pkl"
    preprocessor_artifact_path: Path = MODELS_DIR / "preprocessor.pkl"

    # MLflow Settings
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "Customer_Churn_Prediction"
    registered_model_name: str = "CustomerChurnClassifier"


DEFAULT_CONFIG = ModelConfig()
