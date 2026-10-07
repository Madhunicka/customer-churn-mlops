"""
Production FastAPI Prediction Service for Customer Churn MLOps.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.schemas import (
    BatchCustomerInput,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)
from src.churn.config import DEFAULT_CONFIG
from src.churn.predict import ChurnPredictor

predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads model at startup."""
    global predictor
    try:
        predictor = ChurnPredictor()
        print(f"[INFO] Loaded production model from {DEFAULT_CONFIG.model_artifact_path}")
    except Exception as e:
        print(f"[WARNING] Could not load model at startup: {e}")
        predictor = None
    yield


app = FastAPI(
    title="Customer Churn Prediction MLOps API",
    description=(
        "Production Machine Learning Service for Telco Customer Churn Prediction. "
        "Includes risk scoring, automated retention recommendations, and container health probes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor() -> ChurnPredictor:
    """Helper to ensure predictor is loaded even without lifespan invocation."""
    global predictor
    if predictor is None:
        predictor = ChurnPredictor()
    return predictor


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Liveness & readiness probe for Kubernetes / Docker container health checks."""
    global predictor
    if predictor is None:
        try:
            predictor = ChurnPredictor()
        except Exception:
            pass

    is_loaded = predictor is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        service="customer-churn-mlops",
        version="1.0.0",
        model_loaded=is_loaded,
        model_path=str(DEFAULT_CONFIG.model_artifact_path),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerInput):
    """
    Predict churn for a single customer.
    Returns churn probability, binary prediction (0/1), risk tier, and retention actions.
    """
    try:
        current_predictor = get_predictor()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded. Ensure training artifact exists: {e}",
        )

    customer_dict = customer.model_dump()
    result = current_predictor.predict_single(customer_dict)
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_churn_batch(batch: BatchCustomerInput):
    """
    High-throughput batch inference endpoint for customer datasets.
    """
    try:
        current_predictor = get_predictor()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded. Ensure training artifact exists: {e}",
        )

    records = [c.model_dump() for c in batch.customers]
    predictions = current_predictor.predict_batch(records)
    high_risk = sum(1 for p in predictions if p["risk_tier"] == "High")

    return BatchPredictionResponse(
        total_customers=len(predictions),
        high_risk_count=high_risk,
        predictions=predictions,
    )


@app.get("/", response_class=HTMLResponse, tags=["UI Dashboard"])
def ui_dashboard():
    """Interactive Web Dashboard for interactive model testing."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Customer Churn MLOps Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 31, 51, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: radial-gradient(circle at 15% 15%, #1e1b4b 0%, #0b0f19 50%, #050814 100%);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-height: 100vh;
      padding: 30px 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
    }
    .badge {
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.5px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .badge-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; }
    h1 { font-size: 1.85rem; font-weight: 800; letter-spacing: -0.5px; }
    .subtitle { color: var(--text-muted); font-size: 0.95rem; margin-top: 4px; }
    .nav-links a {
      color: #93c5fd;
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 600;
      margin-left: 18px;
    }
    .grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 24px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .card h2 { font-size: 1.2rem; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 0.8rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
    select, input {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 9px 12px;
      color: #fff;
      font-size: 0.9rem;
      outline: none;
      transition: all 0.2s ease;
    }
    select:focus, input:focus { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25); }
    .btn {
      grid-column: span 2;
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: #fff;
      font-weight: 700;
      font-size: 1rem;
      padding: 13px;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      margin-top: 10px;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    .btn:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5); }
    .btn-presets { display: flex; gap: 10px; margin-bottom: 16px; }
    .btn-preset {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: #cbd5e1;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .btn-preset:hover { background: rgba(255, 255, 255, 0.12); }
    .result-box { display: flex; flex-direction: column; gap: 18px; }
    .gauge-wrapper { text-align: center; padding: 20px; background: rgba(15, 23, 42, 0.5); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); }
    .prob-val { font-size: 3rem; font-weight: 800; }
    .tier-tag { font-size: 0.9rem; font-weight: 700; padding: 4px 14px; border-radius: 99px; display: inline-block; margin-top: 6px; }
    .insights-card { background: rgba(15, 23, 42, 0.6); padding: 16px; border-radius: 12px; border-left: 4px solid var(--primary); }
    .insights-card h3 { font-size: 0.95rem; margin-bottom: 8px; font-weight: 700; }
    .insights-card ul { list-style-type: none; display: flex; flex-direction: column; gap: 6px; }
    .insights-card li { font-size: 0.88rem; color: #cbd5e1; display: flex; align-items: flex-start; gap: 6px; }
    .insights-card li::before { content: "•"; color: var(--primary); font-weight: bold; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <div class="badge"><span class="badge-dot"></span> MLOps Production Engine v1.0</div>
        <h1>Customer Churn Decision System</h1>
        <div class="subtitle">Scikit-learn • MLflow Registry • FastAPI Service • Retention Engine</div>
      </div>
      <div class="nav-links">
        <a href="/docs" target="_blank">Swagger OpenAPI Docs ↗</a>
        <a href="/health" target="_blank">Health Check ↗</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <h2><span>👤</span> Customer Profile Attributes</h2>
        <div class="btn-presets">
          <span style="font-size: 0.8rem; color: var(--text-muted); align-self: center;">Fill Preset:</span>
          <button class="btn-preset" onclick="fillProfile('high')">🚨 High Churn Risk</button>
          <button class="btn-preset" onclick="fillProfile('low')">🛡️ Loyal Customer</button>
        </div>

        <form id="churnForm">
          <div class="form-grid">
            <div class="form-group">
              <label>Gender</label>
              <select id="gender"><option>Male</option><option>Female</option></select>
            </div>
            <div class="form-group">
              <label>Senior Citizen</label>
              <select id="SeniorCitizen"><option value="0">No (0)</option><option value="1">Yes (1)</option></select>
            </div>
            <div class="form-group">
              <label>Partner</label>
              <select id="Partner"><option>Yes</option><option>No</option></select>
            </div>
            <div class="form-group">
              <label>Dependents</label>
              <select id="Dependents"><option>Yes</option><option>No</option></select>
            </div>
            <div class="form-group">
              <label>Tenure (Months)</label>
              <input type="number" id="tenure" value="2" min="0" max="100">
            </div>
            <div class="form-group">
              <label>Contract Type</label>
              <select id="Contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select>
            </div>
            <div class="form-group">
              <label>Monthly Charges ($)</label>
              <input type="number" step="0.01" id="MonthlyCharges" value="85.50">
            </div>
            <div class="form-group">
              <label>Total Charges ($)</label>
              <input type="number" step="0.01" id="TotalCharges" value="171.00">
            </div>
            <div class="form-group">
              <label>Internet Service</label>
              <select id="InternetService"><option>Fiber optic</option><option>DSL</option><option>No</option></select>
            </div>
            <div class="form-group">
              <label>Payment Method</label>
              <select id="PaymentMethod">
                <option>Electronic check</option>
                <option>Mailed check</option>
                <option>Bank transfer (automatic)</option>
                <option>Credit card (automatic)</option>
              </select>
            </div>
            <div class="form-group">
              <label>Tech Support</label>
              <select id="TechSupport"><option>No</option><option>Yes</option><option>No internet service</option></select>
            </div>
            <div class="form-group">
              <label>Online Security</label>
              <select id="OnlineSecurity"><option>No</option><option>Yes</option><option>No internet service</option></select>
            </div>
            <div class="form-group">
              <label>Online Backup</label>
              <select id="OnlineBackup"><option>No</option><option>Yes</option><option>No internet service</option></select>
            </div>
            <div class="form-group">
              <label>Device Protection</label>
              <select id="DeviceProtection"><option>No</option><option>Yes</option><option>No internet service</option></select>
            </div>
            <div class="form-group">
              <label>Streaming TV</label>
              <select id="StreamingTV"><option>No</option><option>Yes</option><option>No internet service</option></select>
            </div>
            <div class="form-group">
              <label>Streaming Movies</label>
              <select id="StreamingMovies"><option>No</option><option>Yes</option><option>No internet service</option></select>
            </div>
            <div class="form-group">
              <label>Phone Service</label>
              <select id="PhoneService"><option>Yes</option><option>No</option></select>
            </div>
            <div class="form-group">
              <label>Multiple Lines</label>
              <select id="MultipleLines"><option>No</option><option>Yes</option><option>No phone service</option></select>
            </div>
            <div class="form-group" style="grid-column: span 2;">
              <label>Paperless Billing</label>
              <select id="PaperlessBilling"><option>Yes</option><option>No</option></select>
            </div>
            <button type="submit" class="btn" id="submitBtn">⚡ Evaluate Churn Risk</button>
          </div>
        </form>
      </div>

      <div class="card result-box">
        <h2><span>📊</span> Prediction & Retention Insights</h2>
        
        <div class="gauge-wrapper">
          <div style="font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Churn Probability</div>
          <div class="prob-val" id="probDisplay">--%</div>
          <div class="tier-tag" id="tierDisplay" style="background: rgba(255,255,255,0.1); color: #fff;">Awaiting Input</div>
        </div>

        <div class="insights-card" id="driversCard" style="border-left-color: #f59e0b;">
          <h3>⚠️ Key Risk Drivers</h3>
          <ul id="driversList">
            <li>Run prediction to compute contributing factors</li>
          </ul>
        </div>

        <div class="insights-card" id="actionsCard" style="border-left-color: #10b981;">
          <h3>📋 Recommended Retention Actions</h3>
          <ul id="actionsList">
            <li>Automated intervention playbook will appear here</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <script>
    function fillProfile(type) {
      if (type === 'high') {
        document.getElementById('tenure').value = 2;
        document.getElementById('Contract').value = 'Month-to-month';
        document.getElementById('MonthlyCharges').value = 95.85;
        document.getElementById('TotalCharges').value = 191.70;
        document.getElementById('InternetService').value = 'Fiber optic';
        document.getElementById('PaymentMethod').value = 'Electronic check';
        document.getElementById('TechSupport').value = 'No';
        document.getElementById('OnlineSecurity').value = 'No';
      } else {
        document.getElementById('tenure').value = 48;
        document.getElementById('Contract').value = 'Two year';
        document.getElementById('MonthlyCharges').value = 24.50;
        document.getElementById('TotalCharges').value = 1176.00;
        document.getElementById('InternetService').value = 'DSL';
        document.getElementById('PaymentMethod').value = 'Credit card (automatic)';
        document.getElementById('TechSupport').value = 'Yes';
        document.getElementById('OnlineSecurity').value = 'Yes';
      }
    }

    document.getElementById('churnForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('submitBtn');
      btn.textContent = 'Processing...';
      btn.disabled = true;

      const payload = {
        gender: document.getElementById('gender').value,
        SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
        Partner: document.getElementById('Partner').value,
        Dependents: document.getElementById('Dependents').value,
        tenure: parseInt(document.getElementById('tenure').value),
        PhoneService: document.getElementById('PhoneService').value,
        MultipleLines: document.getElementById('MultipleLines').value,
        InternetService: document.getElementById('InternetService').value,
        OnlineSecurity: document.getElementById('OnlineSecurity').value,
        OnlineBackup: document.getElementById('OnlineBackup').value,
        DeviceProtection: document.getElementById('DeviceProtection').value,
        TechSupport: document.getElementById('TechSupport').value,
        StreamingTV: document.getElementById('StreamingTV').value,
        StreamingMovies: document.getElementById('StreamingMovies').value,
        Contract: document.getElementById('Contract').value,
        PaperlessBilling: document.getElementById('PaperlessBilling').value,
        PaymentMethod: document.getElementById('PaymentMethod').value,
        MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
        TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
      };

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        const prob = Math.round(data.churn_probability * 100);
        const probElem = document.getElementById('probDisplay');
        const tierElem = document.getElementById('tierDisplay');
        
        probElem.textContent = prob + '%';
        if (data.risk_tier === 'High') {
          probElem.style.color = '#ef4444';
          tierElem.style.background = 'rgba(239, 68, 68, 0.2)';
          tierElem.style.color = '#f87171';
          tierElem.textContent = '🚨 High Churn Risk';
        } else if (data.risk_tier === 'Medium') {
          probElem.style.color = '#f59e0b';
          tierElem.style.background = 'rgba(245, 158, 11, 0.2)';
          tierElem.style.color = '#fbbf24';
          tierElem.textContent = '⚠️ Moderate Churn Risk';
        } else {
          probElem.style.color = '#10b981';
          tierElem.style.background = 'rgba(16, 185, 129, 0.2)';
          tierElem.style.color = '#34d399';
          tierElem.textContent = '🛡️ Low Churn Risk';
        }

        const driversList = document.getElementById('driversList');
        driversList.innerHTML = data.retention_insights.key_risk_drivers.map(d => `<li>${d}</li>`).join('');

        const actionsList = document.getElementById('actionsList');
        actionsList.innerHTML = data.retention_insights.recommended_actions.map(a => `<li>${a}</li>`).join('');
      } catch (err) {
        alert('Prediction failed. Make sure model is trained and API is active.');
      } finally {
        btn.textContent = '⚡ Evaluate Churn Risk';
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)
