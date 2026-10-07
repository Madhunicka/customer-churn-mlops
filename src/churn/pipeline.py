"""
Scikit-Learn Preprocessing & Model Pipeline Definitions.
"""
from pathlib import Path
from typing import Literal, Union
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.churn.config import DEFAULT_CONFIG, ModelConfig


def build_preprocessor(config: ModelConfig = DEFAULT_CONFIG) -> ColumnTransformer:
    """Creates a ColumnTransformer for numerical and categorical features."""
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, config.numerical_features),
            ("cat", categorical_pipeline, config.categorical_features),
        ]
    )

    return preprocessor


def build_model_pipeline(
    model_type: Literal["logistic", "random_forest"] = "logistic",
    config: ModelConfig = DEFAULT_CONFIG,
) -> Pipeline:
    """Builds an end-to-end Scikit-learn Pipeline with preprocessing and classifier."""
    preprocessor = build_preprocessor(config)

    if model_type == "logistic":
        classifier = LogisticRegression(
            C=config.lr_c,
            max_iter=config.lr_max_iter,
            random_state=config.random_state,
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            random_state=config.random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'logistic' or 'random_forest'.")

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", classifier),
        ]
    )

    return pipeline


def save_pipeline(pipeline: Pipeline, output_path: Union[str, Path]) -> Path:
    """Saves a fitted pipeline to disk using joblib."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(model_path: Union[str, Path]) -> Pipeline:
    """Loads a serialized pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)
