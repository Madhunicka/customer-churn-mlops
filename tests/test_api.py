"""
Integration Tests for FastAPI Prediction Endpoints and Schemas.
"""
from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_customer_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20,
    }


def test_health_endpoint():
    """Checks liveness/readiness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "customer-churn-mlops"
    assert "status" in data


def test_ui_dashboard():
    """Checks UI dashboard response."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_single_prediction(sample_customer_payload):
    """Checks single customer prediction output format."""
    response = client.post("/predict", json=sample_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert "churn_prediction" in data
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert "retention_insights" in data
    assert isinstance(data["retention_insights"]["key_risk_drivers"], list)
    assert isinstance(data["retention_insights"]["recommended_actions"], list)


def test_prediction_invalid_schema():
    """Checks that invalid input schema triggers HTTP 422 Unprocessable Entity."""
    invalid_payload = {
        "gender": "Female",
        "tenure": "not-an-integer",  # Invalid type
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_prediction(sample_customer_payload):
    """Checks batch inference endpoint."""
    batch_payload = {
        "customers": [sample_customer_payload, sample_customer_payload]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_customers"] == 2
    assert "high_risk_count" in data
    assert len(data["predictions"]) == 2
