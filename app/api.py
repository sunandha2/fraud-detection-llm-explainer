from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(project_root):
    os.chdir(project_root)

app = FastAPI(
    title="Fraud Detection API",
    description="XGBoost + SHAP fraud detection — predicts fraud probability for any transaction",
    version="1.0.0"
)

# Load model once at startup
model = joblib.load('models/fraud_model.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: str = "W"
    card4: str = "visa"
    card6: str = "debit"
    hour: float = 12.0
    is_night: int = 0
    is_weekend: int = 0
    C1: float = 1.0
    C2: float = 1.0
    C8: float = 1.0
    has_identity: int = 0

class PredictionResponse(BaseModel):
    transaction_amount: float
    fraud_probability: float
    risk_level: str
    recommendation: str

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "model": "XGBoost",
        "roc_auc": 0.90,
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    # Build feature vector
    product_map = {'W': 0, 'C': 1, 'R': 2, 'H': 3, 'S': 4}
    card4_map = {'visa': 0, 'mastercard': 1,
                 'american express': 2, 'discover': 3}
    card6_map = {'credit': 0, 'debit': 1,
                 'debit or credit': 2, 'charge card': 3}

    features = {col: -999 for col in feature_cols}

    features.update({
        'TransactionAmt': transaction.TransactionAmt,
        'amt_log': np.log1p(transaction.TransactionAmt),
        'amt_rounded': int(transaction.TransactionAmt % 1 == 0),
        'amt_cents_99': int(transaction.TransactionAmt % 1 > 0.98),
        'hour': transaction.hour,
        'is_night': transaction.is_night,
        'is_weekend': transaction.is_weekend,
        'day_of_week': 0,
        'product_enc': product_map.get(transaction.ProductCD, 0),
        'card4_enc': card4_map.get(transaction.card4, 0),
        'card6_enc': card6_map.get(transaction.card6, 0),
        'C1': transaction.C1,
        'C2': transaction.C2,
        'C8': transaction.C8,
        'has_identity': transaction.has_identity,
        'p_email_isna': 1,
        'r_email_isna': 1,
        'email_match': 0,
        'p_email_risky': 0,
        'identity_count': 0,
        'is_mobile': 0,
    })

    X = pd.DataFrame([features])[feature_cols].fillna(-999)
    fraud_prob = float(model.predict_proba(X)[0][1])

    if fraud_prob > 0.6:
        risk_level = "HIGH"
        recommendation = "Hold transaction — contact cardholder immediately"
    elif fraud_prob > 0.3:
        risk_level = "MEDIUM"
        recommendation = "Flag for manual review within 1 hour"
    else:
        risk_level = "LOW"
        recommendation = "Allow transaction — standard monitoring"

    return PredictionResponse(
        transaction_amount=transaction.TransactionAmt,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level,
        recommendation=recommendation
    )