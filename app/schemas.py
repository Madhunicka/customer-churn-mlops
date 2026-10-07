"""
Pydantic Schemas for API Request Validation and Response Serialization.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Schema for individual customer prediction requests."""
    gender: str = Field(..., examples=["Female"], description="Gender of customer ('Male' or 'Female')")
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0], description="Whether customer is a senior citizen (1, 0)")
    Partner: str = Field(..., examples=["Yes"], description="Whether customer has a partner ('Yes', 'No')")
    Dependents: str = Field(..., examples=["No"], description="Whether customer has dependents ('Yes', 'No')")
    tenure: int = Field(..., ge=0, examples=[12], description="Number of months customer has stayed with company")
    PhoneService: str = Field(..., examples=["Yes"], description="Whether customer has phone service ('Yes', 'No')")
    MultipleLines: str = Field(..., examples=["No"], description="Whether customer has multiple lines")
    InternetService: str = Field(..., examples=["DSL"], description="Customer's internet service provider ('DSL', 'Fiber optic', 'No')")
    OnlineSecurity: str = Field(..., examples=["No"], description="Whether customer has online security")
    OnlineBackup: str = Field(..., examples=["Yes"], description="Whether customer has online backup")
    DeviceProtection: str = Field(..., examples=["No"], description="Whether customer has device protection")
    TechSupport: str = Field(..., examples=["No"], description="Whether customer has tech support")
    StreamingTV: str = Field(..., examples=["No"], description="Whether customer has streaming TV")
    StreamingMovies: str = Field(..., examples=["No"], description="Whether customer has streaming movies")
    Contract: str = Field(..., examples=["Month-to-month"], description="Contract term ('Month-to-month', 'One year', 'Two year')")
    PaperlessBilling: str = Field(..., examples=["Yes"], description="Whether customer has paperless billing ('Yes', 'No')")
    PaymentMethod: str = Field(..., examples=["Electronic check"], description="Payment method used")
    MonthlyCharges: float = Field(..., gt=0, examples=[53.85], description="Monthly amount charged to customer")
    TotalCharges: float = Field(..., ge=0, examples=[646.2], description="Total amount charged to customer")


class BatchCustomerInput(BaseModel):
    """Schema for batch inference requests."""
    customers: List[CustomerInput]


class RetentionInsights(BaseModel):
    key_risk_drivers: List[str]
    recommended_actions: List[str]
    intervention_priority: str


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_label: str
    churn_probability: float
    risk_tier: str
    retention_insights: RetentionInsights


class BatchPredictionResponse(BaseModel):
    total_customers: int
    high_risk_count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    model_path: str
