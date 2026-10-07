"""
Unit Tests for Scikit-Learn Preprocessing and Model Pipelines.
"""
from pathlib import Path
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.churn.config import DEFAULT_CONFIG
from src.churn.dataset import get_train_test_splits
from src.churn.pipeline import build_model_pipeline, build_preprocessor, load_pipeline, save_pipeline


def test_build_preprocessor():
    """Verifies preprocessor construction with ColumnTransformer."""
    preprocessor = build_preprocessor(DEFAULT_CONFIG)
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2
    assert preprocessor.transformers[0][0] == "num"
    assert preprocessor.transformers[1][0] == "cat"


def test_model_pipeline_fit_and_predict():
    """Verifies end-to-end pipeline training and prediction probabilities."""
    X_train, X_test, y_train, y_test = get_train_test_splits()

    # Train with small subset for test speed
    X_train_sub = X_train.iloc[:200]
    y_train_sub = y_train.iloc[:200]
    X_test_sub = X_test.iloc[:50]

    pipeline = build_model_pipeline("logistic", DEFAULT_CONFIG)
    pipeline.fit(X_train_sub, y_train_sub)

    predictions = pipeline.predict(X_test_sub)
    probabilities = pipeline.predict_proba(X_test_sub)

    assert len(predictions) == len(X_test_sub)
    assert probabilities.shape == (len(X_test_sub), 2)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_save_and_load_pipeline(tmp_path: Path):
    """Verifies pipeline serialization and deserialization via joblib."""
    X_train, X_test, y_train, y_test = get_train_test_splits()
    pipeline = build_model_pipeline("logistic", DEFAULT_CONFIG)
    pipeline.fit(X_train.iloc[:100], y_train.iloc[:100])

    model_path = tmp_path / "test_model.pkl"
    save_pipeline(pipeline, model_path)
    assert model_path.exists()

    loaded_pipe = load_pipeline(model_path)
    assert isinstance(loaded_pipe, Pipeline)

    sample = X_test.iloc[:5]
    orig_preds = pipeline.predict(sample)
    loaded_preds = loaded_pipe.predict(sample)
    np.testing.assert_array_equal(orig_preds, loaded_preds)
