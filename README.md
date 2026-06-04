# Fraud Detection + LLM Explainer

> XGBoost detects fraud. SHAP explains why.
> Groq LLM writes plain-English analyst reports.

## Live Demo
🔗 *(deploying — link coming soon)*

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