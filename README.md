# Fraud Detection + LLM Explainer

> XGBoost detects fraud. SHAP explains why. Groq LLM writes compliance-ready analyst reports.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b)](https://fraud-detection-llm-explainer-ivvbpnmpvrfwhuy2mlyhjz.streamlit.app/)
[![API](https://img.shields.io/badge/REST%20API-Render-46E3B7)](https://fraud-detection-llm-explainer.onrender.com/health)
[![Dataset](https://img.shields.io/badge/Dataset-IEEE--CIS%20Kaggle-20BEFF)](https://www.kaggle.com/c/ieee-fraud-detection)

---

## The Problem

Fraud detection models flag transactions — but compliance analysts can't act on "73% fraud probability."

They need to know **WHY** — in plain English, with evidence they can put in an audit trail.

| Without This System | With This System |
|---|---|
| Black-box score, no context | Plain-English fraud pattern report |
| Analyst spends 15 mins per case | Report generated in seconds |
| No audit trail for regulators | Structured evidence: pattern + action |
| False positives block good customers | SHAP explains exactly what triggered the flag |

---

## Live Demo

🔗 **Streamlit Dashboard**: https://fraud-detection-llm-explainer-ivvbpnmpvrfwhuy2mlyhjz.streamlit.app/

🔗 **REST API**: `GET /health` → `{"status": "ok", "model_loaded": true, "features": 45}`

---

## What It Does

1. **Detects fraud** using XGBoost trained on 590,540 real IEEE-CIS transactions
2. **Explains each decision** using SHAP — identifies which features drove the fraud flag per transaction
3. **Classifies the fraud pattern** using Groq LLM (Llama 3.3) — distinguishes card testing vs velocity fraud vs organized fraud ring
4. **Generates analyst reports** — structured output with evidence, confidence, and recommended action
5. **Serves predictions** via a deployed REST API on Render

---

## Example LLM Output

```
Transaction 3405227 — $73.95 — Product C — 98.1% risk

PATTERN: Velocity fraud — stolen card being rapidly tested

EVIDENCE:
  • C1=108 (normal < 5): card used 108 times — abnormal velocity
  • C8=100: transaction count anomaly confirms repeat usage pattern
  • is_night=1: transaction at 2am — outside cardholder's normal hours
  • ProductCD=C: 11.7% base fraud rate — highest risk product category

CONFIDENCE: High — C1 > 50 AND C8 > 50 simultaneously is the primary
stolen card signal. Night timing + Product C combination confirms organized pattern.

ACTION: Block card immediately. Contact cardholder for verification.
Flag for compliance review — velocity pattern warrants investigation of
linked transactions.
```

The LLM doesn't just format data — it **reasons conditionally**:
- If C1 > 50 AND C8 > 50 simultaneously → velocity fraud classification
- If is_night=1 AND ProductCD=C AND high amount → organized fraud ring pattern
- If single high-value transaction, no velocity → card testing classification

An f-string template cannot do this. The LLM chooses the pattern based on which combination of features is present.

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | **0.90** |
| High-risk precision | **88.2%** |
| High-risk transactions flagged | 7,355 |
| Amount protected | $846,978 |
| Analyst time saved | ~15 mins per transaction |

### Why ROC-AUC Matters Here

With 96.5% legitimate transactions, a dumb model that predicts "always legitimate" gets 96.5% accuracy but catches **zero fraud**. ROC-AUC measures the model's ability to rank fraud above legitimate — independent of class imbalance.

### Model Comparison

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Always predict legitimate (baseline) | 0.50 | 0.00 | 0.00 | 0.00 |
| Logistic Regression | 0.72 | 0.59 | 0.10 | 0.18 |
| Random Forest | 0.84 | 0.70 | 0.19 | 0.30 |
| **XGBoost (final)** | **0.90** | **0.77** | **0.27** | **0.40** |

XGBoost is **+0.18 ROC-AUC over Logistic Regression** and **+0.22 over the dumb baseline**.

### SMOTE Impact

| | ROC-AUC |
|---|---|
| XGBoost without SMOTE | 0.86 |
| XGBoost with SMOTE | **0.90** |
| Improvement | **+0.04** |

**Critical**: SMOTE was applied **only on the training split**, never on the test set. Applying SMOTE before splitting causes data leakage — synthetic samples from the same original transactions appear in both train and test, inflating metrics artificially. Test set kept the original 96.5%/3.5% distribution to reflect real-world performance.

---

## Class Imbalance — The Core Challenge

```
Total transactions:  590,540
Fraudulent:           20,663  (3.5%)
Legitimate:          569,877  (96.5%)
```

A model that predicts "legitimate" for everything gets 96.5% accuracy and catches zero fraud. This is the **accuracy paradox** — the core reason naive models fail on fraud data.

Solution: SMOTE (Synthetic Minority Oversampling Technique) on training data only, with `sampling_strategy=0.1` to avoid over-correcting.

---

## SHAP Insights — What Actually Drives Fraud

SHAP (SHapley Additive exPlanations) calculates each feature's contribution to every individual prediction. Key findings:

**1. Time beats amount**
Time signals are 2x more predictive than transaction amount. WHEN a transaction happens matters more than HOW MUCH it is. Most people assume large amounts = fraud. The model learned otherwise.

**2. C-fields dominate (24% of all model decisions)**
Transaction count fields (C1–C10) explain 24% of all decisions. Stolen cards get used repeatedly — this is the primary fraud signal. C1 (how many times the card has been used) is the single most important feature.

**3. Email mismatch misleads**
83% of LEGITIMATE transactions have mismatched email domains. Email mismatch is normal behavior, not a fraud signal. SHAP correctly learned to ignore it — this is non-obvious and counterintuitive.

**4. Product C is structurally high risk**
Product C is 4.4x more likely to be flagged than Product W. This aligns with the raw data: 11.69% fraud rate for Product C vs 2.04% for Product W.

**5. Night transactions**
`is_night` ranked #3 globally across all SHAP values. Fraudsters operate between 10pm–6am when victims are asleep and less likely to notice alerts.

---

## Feature Engineering

45 features engineered from raw transaction + identity data:

| Feature | Logic | Why |
|---|---|---|
| `is_night` | hour < 6 or hour > 22 | Fraudster operating hours |
| `amt_log` | log1p(TransactionAmt) | Normalizes skewed distribution |
| `amt_cents_99` | amount % 1 > 0.98 | Classic .99 pricing fraud signal |
| `email_match` | P_email == R_email | Billing/shipping domain consistency |
| `p_email_risky` | domain in risky list | Anonymous/throwaway email detection |
| `identity_count` | count of non-null id_ fields | Identity completeness as trust signal |
| `card4_enc` | visa/mc/amex/discover encoded | Card network risk profile |
| `C1`–`C10` | raw Vesta count features | Transaction velocity per card/address |
| `D1`–`D4` | raw Vesta timedelta features | Days since last transaction |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| XGBoost | Fraud classification — gradient boosting on tabular data |
| SHAP TreeExplainer | Per-transaction feature attribution (O(n) complexity) |
| Groq API (Llama 3.3-70B) | Conditional fraud pattern classification + report generation |
| SMOTE (imbalanced-learn) | Minority class oversampling on training data only |
| Flask + Gunicorn | REST API serving |
| Render | API deployment |
| Streamlit | Live analyst dashboard |

---

## REST API

Deployed on Render. Accepts transaction data, returns fraud probability + risk level.

Deployed on Render: https://fraud-detection-llm-explainer.onrender.com

Accepts transaction data, returns fraud probability + risk level.

**Endpoints:**
- `GET /health` — confirms model loaded and feature count
- `POST /predict` — returns fraud score
- `GET /features` — lists all 45 features

**Example request:**
```json
POST /predict
{
  "TransactionAmt": 450.00,
  "ProductCD": "C",
  "card4": "discover",
  "hour": 2.0,
  "is_night": 1,
  "C1": 108,
  "C8": 100
}
```

**Example response:**
```json
{
  "fraud_probability": 0.94,
  "risk_level": "High",
  "is_fraud": true
}
```

---

## Dataset

**IEEE-CIS Fraud Detection** — Kaggle competition dataset from Vesta Corporation (real e-commerce payment processor)

- 590,540 real financial transactions
- 20,663 fraudulent (3.5% fraud rate)
- 434 raw features → 45 engineered features used
- Transaction + identity tables joined on TransactionID

---

## Project Structure

```
fraud-detection-llm-explainer/
├── app.py                      # Flask REST API
├── requirements.txt
├── render.yaml                 # Render deployment config
├── notebooks/
│   ├── explore.py              # EDA — fraud rate, product analysis, missing values
│   ├── train_model.py          # XGBoost training + SMOTE + evaluation
│   ├── model_comparison.py     # Logistic Regression vs RF vs XGBoost
│   ├── shap_analysis.py        # SHAP values + per-transaction explanations
│   └── shap_insights.py        # 5 key SHAP findings
├── models/
│   ├── fraud_model.pkl
│   └── feature_cols.pkl
├── outputs/
│   ├── fraud_predictions.csv
│   ├── shap_values.csv
│   ├── shap_global_importance.csv
│   ├── model_comparison.csv
│   └── smote_comparison.csv
└── app/
    └── main.py                 # Streamlit dashboard
```

---

## How to Run

```bash
git clone https://github.com/sunandha2/fraud-detection-llm-explainer
cd fraud-detection-llm-explainer
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Train model (requires data/train_transaction.csv + data/train_identity.csv)
python notebooks/train_model.py

# Run Streamlit dashboard
streamlit run app/main.py

# Run REST API locally
python app.py
```

---

## Resume Bullet

**Fraud Detection & Explainability System** — XGBoost classifier on 590K real IEEE-CIS transactions (ROC-AUC 0.90 vs 0.72 Logistic Regression baseline); addressed 96.5% class imbalance via SMOTE applied exclusively on training split to prevent data leakage

**SHAP per-transaction explainability** + Groq LLM (Llama 3.3) conditional pattern classifier distinguishes card testing vs velocity fraud vs organized fraud ring; Flask REST API deployed on Render; $846,978 in high-risk transactions flagged across 7,355 transactions

---

## Author

Built to demonstrate end-to-end ML + explainability + LLM integration for financial fraud detection.