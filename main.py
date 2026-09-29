# api/main.py
# FastAPI deployment for Loan Default Prediction Model
# Loads the trained XGBoost model and serves predictions
# via a REST API endpoint

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pickle
import pandas as pd
import numpy as np
import uvicorn
import os

# ── LOAD MODEL ──────────────────────────────────────
print("Loading loan default prediction model...")
try:
    with open("models/loan_model.pkl", "rb") as f:
        artifacts = pickle.load(f)

    model    = artifacts["model"]
    scaler   = artifacts["scaler"]
    imputer  = artifacts["imputer"]
    encoders = artifacts["encoders"]
    features = artifacts["features"]
    print("  Model loaded successfully.")
    print(f"  Features expected: {len(features)}")
except Exception as e:
    print(f"  Model load failed: {e}")
    model = None

# ── APP ─────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Predicts whether a loan applicant will default "
        "using XGBoost trained on Home Credit data."
    ),
    version="1.0.0"
)

# ── REQUEST SCHEMA ───────────────────────────────────
class LoanApplication(BaseModel):
    AMT_CREDIT:        float
    AMT_INCOME_TOTAL:  float
    AMT_ANNUITY:       float
    DAYS_BIRTH:        int
    DAYS_EMPLOYED:     int
    EXT_SOURCE_1:      Optional[float] = None
    EXT_SOURCE_2:      Optional[float] = None
    EXT_SOURCE_3:      Optional[float] = None
    NAME_CONTRACT_TYPE:   Optional[str] = "Cash loans"
    NAME_EDUCATION_TYPE:  Optional[str] = "Secondary / secondary special"
    NAME_FAMILY_STATUS:   Optional[str] = "Married"
    NAME_INCOME_TYPE:     Optional[str] = "Working"
    FLAG_OWN_CAR:         Optional[str] = "N"
    FLAG_OWN_REALTY:      Optional[str] = "Y"
    CNT_FAM_MEMBERS:      Optional[float] = 2.0
    REGION_RATING_CLIENT: Optional[int] = 2


def prepare_input(data: LoanApplication) -> pd.DataFrame:
    """
    Converts API input into the same feature format
    the model was trained on — including all engineered
    features from Section 4.
    """
    raw = {
        "AMT_CREDIT":           data.AMT_CREDIT,
        "AMT_INCOME_TOTAL":     data.AMT_INCOME_TOTAL,
        "AMT_ANNUITY":          data.AMT_ANNUITY,
        "DAYS_BIRTH":           data.DAYS_BIRTH,
        "DAYS_EMPLOYED":        data.DAYS_EMPLOYED,
        "EXT_SOURCE_1":         data.EXT_SOURCE_1,
        "EXT_SOURCE_2":         data.EXT_SOURCE_2,
        "EXT_SOURCE_3":         data.EXT_SOURCE_3,
        "NAME_CONTRACT_TYPE":   data.NAME_CONTRACT_TYPE,
        "NAME_EDUCATION_TYPE":  data.NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS":   data.NAME_FAMILY_STATUS,
        "NAME_INCOME_TYPE":     data.NAME_INCOME_TYPE,
        "FLAG_OWN_CAR":         data.FLAG_OWN_CAR,
        "FLAG_OWN_REALTY":      data.FLAG_OWN_REALTY,
        "CNT_FAM_MEMBERS":      data.CNT_FAM_MEMBERS,
        "REGION_RATING_CLIENT": data.REGION_RATING_CLIENT,
    }
    df = pd.DataFrame([raw])

    # Fix DAYS_EMPLOYED anomaly
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Engineer features — same as Section 4
    df["AGE_YEARS"]      = (-df["DAYS_BIRTH"]    / 365).round(1)
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).round(1)

    df["CREDIT_INCOME_RATIO"]  = (
        df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)).round(4)
    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)).round(4)
    df["CREDIT_TERM"]          = (
        df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)).round(1)
    df["EMPLOYED_TO_AGE_RATIO"] = (
        df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)).round(4)
    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)).round(2)

    ext_cols  = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available = [c for c in ext_cols if c in df.columns]
    if available:
        df["EXT_SOURCE_MEAN"] = df[available].mean(axis=1).round(4)
        df["EXT_SOURCE_MIN"]  = df[available].min(axis=1).round(4)
        df["EXT_SOURCE_MAX"]  = df[available].max(axis=1).round(4)

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].apply(
                lambda x: x if x in le.classes_ else "Unknown")
            df[col] = le.transform(df[col])

    # Align columns to training feature set
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    # Impute and scale
    df_imputed = imputer.transform(df)
    df_scaled  = scaler.transform(df_imputed)

    return df_scaled


def get_risk_level(probability: float) -> dict:
    """
    Converts raw probability into a human-readable
    risk level and recommendation.
    """
    if probability < 0.3:
        return {
            "risk_level":     "LOW",
            "recommendation": "APPROVE",
            "explanation":    (
                "Low probability of default. "
                "Applicant profile is financially stable."
            )
        }
    elif probability < 0.6:
        return {
            "risk_level":     "MEDIUM",
            "recommendation": "REVIEW",
            "explanation":    (
                "Moderate default risk. "
                "Manual review recommended before approval."
            )
        }
    else:
        return {
            "risk_level":     "HIGH",
            "recommendation": "REJECT",
            "explanation":    (
                "High probability of default. "
                "Loan approval not recommended."
            )
        }


# ── ENDPOINTS ────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "service":      "Loan Default Prediction API",
        "version":      "1.0.0"
    }


@app.post("/predict")
def predict(application: LoanApplication):
    """
    Submit a loan application and receive a
    default risk prediction with recommendation.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the notebook first."
        )
    try:
        input_data   = prepare_input(application)
        probability  = float(
            model.predict_proba(input_data)[0][1])
        risk         = get_risk_level(probability)

        return {
            "default_probability": round(probability, 4),
            "risk_level":          risk["risk_level"],
            "recommendation":      risk["recommendation"],
            "explanation":         risk["explanation"],
            "model":               "XGBoost",
            "version":             "1.0.0"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(applications: list[LoanApplication]):
    """
    Submit multiple loan applications at once.
    Returns predictions for all in one response.
    """
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded.")
    results = []
    for i, app in enumerate(applications):
        try:
            input_data  = prepare_input(app)
            probability = float(
                model.predict_proba(input_data)[0][1])
            risk        = get_risk_level(probability)
            results.append({
                "application_index":   i,
                "default_probability": round(probability, 4),
                "risk_level":          risk["risk_level"],
                "recommendation":      risk["recommendation"],
            })
        except Exception as e:
            results.append({
                "application_index": i,
                "error":             str(e)
            })
    return {"total": len(results), "predictions": results}


@app.get("/stats")
def stats():
    """
    Returns model information and threshold settings.
    """
    return {
        "model":           "XGBoost Classifier",
        "trained_on":      "Home Credit Default Risk — Kaggle",
        "features_used":   len(features) if features else 0,
        "risk_thresholds": {
            "LOW":    "probability < 0.30  → APPROVE",
            "MEDIUM": "probability 0.30 to 0.60 → REVIEW",
            "HIGH":   "probability > 0.60  → REJECT",
        },
        "target_metric":  "AUC-ROC"
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Default Prediction API</title>
    <style>
        body { font-family: Arial, sans-serif;
               max-width: 700px; margin: 60px auto;
               padding: 20px; background: #f5f5f5; }
        h1   { color: #1a237e; }
        h3   { color: #3949ab; }
        .box { background: white; padding: 20px;
               border-radius: 10px; margin: 16px 0;
               border-left: 4px solid #1a237e; }
        a    { color: #1a237e; font-weight: bold; }
        code { background: #eef2ff; padding: 2px 6px;
               border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Loan Default Prediction API</h1>
    <p>XGBoost model trained on Home Credit Default Risk data.</p>

    <div class="box">
        <h3>Available Endpoints</h3>
        <p><code>GET  /health</code> — API status</p>
        <p><code>POST /predict</code> — Single application prediction</p>
        <p><code>POST /predict/batch</code> — Multiple applications</p>
        <p><code>GET  /stats</code> — Model information</p>
        <p><a href="/docs">📄 Open API Documentation →</a></p>
    </div>

    <div class="box">
        <h3>Risk Levels</h3>
        <p>🟢 <strong>LOW</strong> — Probability below 0.30 → APPROVE</p>
        <p>🟡 <strong>MEDIUM</strong> — Probability 0.30 to 0.60 → REVIEW</p>
        <p>🔴 <strong>HIGH</strong> — Probability above 0.60 → REJECT</p>
    </div>

    <div class="box">
        <h3>Author</h3>
        <p>Habib Bashir Lawal</p>
        <p>Junior Data Scientist</p>
        <p>IU International University of Applied Sciences, Germany</p>
    </div>
</body>
</html>
    """


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",
                port=8004, reload=True)