# Fraud Detection + LLM Explainer

> XGBoost detects fraud. SHAP explains why.
> Groq LLM writes plain-English analyst reports.

## Live Demo
🔗 https://fraud-detection-llm-explainer-ivvbpnmpvrfwhuy2mlyhjz.streamlit.app/

## The Problem
Fraud detection models flag transactions but compliance
analysts can't act on "73% fraud probability."
They need to know WHY — in plain English.

Without explainability:
- Analysts can't prioritize which flags to investigate
- False positives block legitimate customers
- Regulators demand audit trails for every decision

## What It Does
- Detects fraud using XGBoost on 590,540 real transactions
- Uses SHAP to identify which features drove each fraud flag
- Uses Groq LLM (Llama 3.3) to convert SHAP values into
  plain-English analyst reports
- Live 4-page Streamlit dashboard

## Example LLM Output
Transaction 3405227 — $73.95 — Product C — 98.1% risk

RISK ASSESSMENT: Critical risk — stolen card pattern
detected. Card used 108 times (C1=108), far above normal.

KEY EVIDENCE: C1=108 and C8=100 confirm abnormal
transaction velocity. Product C has 11.7% base fraud rate.

RECOMMENDED ACTION: Block card and contact cardholder
immediately for verification.

## Tech Stack
| Tool | Purpose |
|---|---|
| XGBoost | Fraud classification model |
| SHAP | Per-transaction explainability |
| Groq API (Llama 3.3) | Plain-English report generation |
| SMOTE | Class imbalance handling |
| Streamlit | Live analyst dashboard |

## Model Performance
- ROC-AUC: 0.90
- High risk precision: 88.2%
- High risk transactions: 7,355
- Amount protected: $846,978
- Analyst time saved: ~15 mins per transaction

## Dataset
IEEE-CIS Fraud Detection (Kaggle)
- 590,540 real financial transactions
- 20,663 fraudulent (3.5% fraud rate)
- 45 engineered features
- Real e-commerce payment data

## Model Comparison
| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.88 | 0.77 | 0.27 | 0.40 |
| Random Forest | 0.84 | 0.70 | 0.19 | 0.30 |
| Logistic Regression | 0.72 | 0.59 | 0.10 | 0.18 |

XGBoost won on every metric. Key reasons:
- Handles class imbalance better than Logistic Regression
- Captures non-linear transaction patterns
- Native SHAP compatibility
- Fastest inference on large datasets

## SHAP Key Insights
1. **Time beats amount**: Time signals are 2x more predictive
   than transaction amount — WHEN matters more than HOW MUCH
2. **C-fields dominate**: Transaction count fields explain 24%
   of all model decisions — stolen cards get used repeatedly
3. **Email mismatch misleads**: 83% of LEGITIMATE transactions
   have mismatched email domains — model correctly ignores this
4. **Product C structural risk**: 4.4x more likely flagged
   than Product W — aligns with 11.69% raw fraud rate
5. **Night transactions**: is_night ranked #3 globally —
   fraudsters operate when victims sleep

## REST API
The model is served as a REST API using FastAPI:
- GET /health — health check
- POST /predict — returns fraud probability + recommendation
- GET /docs — interactive API documentation

Example request:
```json
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

Example response:
```json
{
  "fraud_probability": 0.5168,
  "risk_level": "MEDIUM",
  "recommendation": "Flag for manual review within 1 hour"
}
```

## Key Findings
- Product C: 11.69% fraud rate — highest risk
- Discover cards: 7.73% fraud rate
- Top fraud driver: Transaction count anomalies (C1, C8)
- Night transactions significantly higher risk

## Progress
- [x] Day 1 — 590,540 transactions explored (3.5% fraud rate, product C = 11.69% fraud)
- [x] Day 2 — XGBoost trained (ROC-AUC 0.90, 88.2% precision) — 7,355 high-risk transactions flagged
- [x] Day 3 — SHAP explainability — top fraud driver: C1/C8 transaction count anomalies
- [x] Day 4 — Groq LLM analyst reports — 100% precision on top flagged transactions
- [x]Day 5 — Streamlit app + deployment

## How to Run
```bash
git clone https://github.com/sunandha2/fraud-detection-llm-explainer
cd fraud-detection-llm-explainer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python notebooks/train_model.py
streamlit run app/main.py
```

## Resume Bullet
"Fraud detection system on 590K transactions (XGBoost,
ROC-AUC 0.90, 88.2% precision) with SHAP + Groq LLM
explainability — converts model decisions into plain-English
analyst reports. $846,978 in fraudulent transactions
identified. Live Streamlit dashboard deployed."

## Author
Built to demonstrate end-to-end ML + explainability +
LLM integration for financial fraud detection.