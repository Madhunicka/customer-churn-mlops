"""
Inference Engine and Retention Recommendation Service.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from sklearn.pipeline import Pipeline

from src.churn.config import DEFAULT_CONFIG, ModelConfig
from src.churn.pipeline import load_pipeline


class ChurnPredictor:
    """Production predictor wrapper with risk scoring and retention recommendations."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: ModelConfig = DEFAULT_CONFIG,
    ):
        self.config = config
        path = model_path or config.model_artifact_path
        self.pipeline: Pipeline = load_pipeline(path)

    def predict_single(self, customer_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Runs inference for a single customer payload."""
        df = pd.DataFrame([customer_dict])
        return self._format_prediction(df)[0]

    def predict_batch(self, customer_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runs inference for a batch of customer payloads."""
        df = pd.DataFrame(customer_dicts)
        return self._format_prediction(df)

    def _format_prediction(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extracts predictions, probabilities, risk tiers, and retention insights."""
        predictions = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)[:, 1]

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            row = df.iloc[i].to_dict()
            prob_float = float(round(prob, 4))
            churn_int = int(pred)

            if prob_float >= 0.70:
                risk_tier = "High"
            elif prob_float >= 0.40:
                risk_tier = "Medium"
            else:
                risk_tier = "Low"

            # Generate smart retention advice based on key churn drivers
            retention_advice = self._generate_retention_advice(row, risk_tier, prob_float)

            results.append({
                "churn_prediction": churn_int,
                "churn_label": "Yes" if churn_int == 1 else "No",
                "churn_probability": prob_float,
                "risk_tier": risk_tier,
                "retention_insights": retention_advice,
            })

        return results

    @staticmethod
    def _generate_retention_advice(
        customer: Dict[str, Any], risk_tier: str, probability: float
    ) -> Dict[str, Any]:
        """
        Analyzes customer risk drivers and recommends retention actions
        based on account attributes and tenure.
        """
        drivers = []
        actions = []

        contract = customer.get("Contract", "")
        tenure = customer.get("tenure", 0)
        monthly_charges = customer.get("MonthlyCharges", 0.0)
        tech_support = customer.get("TechSupport", "")
        internet_service = customer.get("InternetService", "")
        payment_method = customer.get("PaymentMethod", "")

        if contract == "Month-to-month":
            drivers.append("Short-term month-to-month contract (high flight risk)")
            actions.append("Offer 15% discount for upgrading to 1-Year or 2-Year contract")

        if tenure is not None and float(tenure) <= 12:
            drivers.append("New customer tenure under 1 year (critical onboarding period)")
            actions.append("Enroll in VIP customer success check-in program")

        if monthly_charges is not None and float(monthly_charges) > 75:
            drivers.append(f"Elevated monthly billing (${monthly_charges:.2f}/mo)")
            actions.append("Review plan features and propose tailored bundle discount")

        if tech_support == "No" and internet_service != "No":
            drivers.append("No active tech support subscription")
            actions.append("Provide 3 months complimentary Premium Tech Support")

        if payment_method == "Electronic check":
            drivers.append("Electronic check payment (correlated with higher payment friction)")
            actions.append("Incentivize setup of automated credit card or bank transfer billing")

        if not drivers:
            drivers.append("Healthy engagement indicators; low churn signals")
            actions.append("Maintain standard loyalty engagement and newsletters")

        return {
            "key_risk_drivers": drivers,
            "recommended_actions": actions,
            "intervention_priority": "Immediate" if risk_tier == "High" else "Monitor",
        }
