# Fraud Detection + LLM Explainer

> XGBoost detects fraud. SHAP explains why. 
> Groq LLM translates it into plain English for analysts.

## The Problem
Fraud detection models flag transactions but compliance
analysts can't act on "73% fraud probability."
They need to know WHY — in plain English.

Without explainability:
- Analysts can't prioritize which flags to investigate
- False positives block legitimate customers
- Regulators demand audit trails for every decision

This project solves all three.

## What It Does
- Detects fraud using XGBoost on 590,540 real transactions
- Uses SHAP to identify which features drove each fraud flag
- Uses Groq LLM (Llama 3.3) to convert SHAP values into
  plain-English analyst reports
- Live Streamlit dashboard — paste a transaction, get a report

## Example Output
Input: TransactionAmt=849, ProductCD=C, card4=discover,
       hour=2am, email_match=False

LLM Report:
"This transaction was flagged as HIGH RISK (89% probability)
for three reasons: (1) Product category C has an 11.7% base
fraud rate — 3x the average. (2) The transaction occurred at
2am, which is statistically associated with fraudulent activity.
(3) The sender and receiver email domains don't match,
suggesting a mismatch between accounts. Recommended action:
hold for manual review before processing."

## Tech Stack
| Tool | Purpose |
|---|---|
| XGBoost | Fraud classification model |
| SHAP | Feature importance per transaction |
| Groq API (Llama 3.3) | Plain-English explanation generation |
| Streamlit | Live analyst dashboard |
| SMOTE | Class imbalance handling |
| Python + Pandas | Data pipeline |

## Dataset
IEEE-CIS Fraud Detection (Kaggle)
- 590,540 real financial transactions
- 20,663 fraudulent (3.5% fraud rate)
- 434 features (transactions + identity)
- Real e-commerce payment data

## Key Findings
- Product C has 11.69% fraud rate — highest risk category
- Discover cards have 7.73% fraud rate vs 3.48% for Visa
- Night transactions (10pm-6am) are significantly higher risk
- Email domain mismatch is a strong fraud signal

## Progress
- [x] Day 1 — 590,540 transactions explored (3.5% fraud rate, product C = 11.69% fraud)
- [x] Day 2 — XGBoost trained (ROC-AUC 0.90, 88.2% precision) — 7,355 high-risk transactions flagged
- [x] Day 3 — SHAP explainability — top fraud driver: C1/C8 transaction count anomalies
- [x] Day 4 — Groq LLM analyst reports — 100% precision on top flagged transactions
- [ ] Day 5 — Streamlit app + deployment

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

## Author
Built to demonstrate end-to-end ML + explainability + LLM
integration for financial fraud detection.