import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\fraud-detection-llm-explainer')

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

shap_values = pd.read_csv('outputs/shap_values.csv')
predictions = pd.read_csv('outputs/fraud_predictions.csv')
feature_cols = joblib.load('models/feature_cols.pkl')

print(f"SHAP values: {shap_values.shape}")
print(f"Predictions: {predictions.shape}")

# ── RAG CONTEXT ───────────────────────────────────────────────
RAG_CONTEXT = """
You are a senior fraud analyst at a financial institution.

PRODUCT RISK LEVELS:
- Product C: 11.69% fraud rate — HIGHEST RISK
- Product S: 5.90% fraud rate — HIGH RISK
- Product H: 4.77% fraud rate — MEDIUM-HIGH RISK
- Product R: 3.78% fraud rate — MEDIUM RISK
- Product W: 2.04% fraud rate — LOWER RISK

CARD TYPE FRAUD RATES:
- Discover: 7.73% fraud rate — highest
- Visa: 3.48% fraud rate
- Mastercard: 3.43% fraud rate
- American Express: 2.87% fraud rate — lowest

KEY FRAUD SIGNALS:
- C1, C2, C8 (transaction count fields): High values mean
  the card/address has been used many times — stolen card signal
- is_night (10pm-6am): Fraudsters operate when victims sleep
- is_weekend: Reduced fraud monitoring on weekends
- TransactionAmt: Unusually high amounts are suspicious
- email_match=0: Sender and receiver email domains differ

RISK THRESHOLDS:
- Above 80%: CRITICAL — hold immediately, contact cardholder
- 60-80%: HIGH — flag for manual review within 1 hour
- 30-60%: MEDIUM — monitor, review within 24 hours
- Below 30%: LOW — allow, standard monitoring

RECOMMENDED ACTIONS BY PATTERN:
- High C1/C8 counts: Stolen card — block card, contact cardholder
- Night + high amount: Card testing pattern — temporary hold
- Product C + high probability: High-risk category — manual review
- Email mismatch: Account takeover signal — verify identity
"""

print("\n" + "=" * 60)
print("STEP 2 — Building explanation function")
print("=" * 60)

def get_shap_factors(transaction_id, top_n=5):
    """Get top SHAP factors for a transaction"""
    row = shap_values[
        shap_values['TransactionID'] == transaction_id
    ]
    if row.empty:
        return []

    row = row.iloc[0]
    shap_cols = [c for c in shap_values.columns
                 if c.startswith('shap_')]

    factors = []
    for col in shap_cols:
        feature_name = col.replace('shap_', '')
        shap_val = row[col]
        # Get actual feature value from shap_values
        factors.append({
            'feature': feature_name,
            'shap_value': float(shap_val),
            'direction': 'INCREASES' if shap_val > 0 else 'decreases'
        })

    factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return factors[:top_n]

def generate_fraud_report(transaction_data, shap_factors):
    """Use Groq LLM to generate plain-English fraud analyst report"""

    # Format SHAP factors
    shap_text = "\n".join([
        f"  - {f['feature']}: {f['direction']} fraud risk "
        f"(impact: {abs(f['shap_value']):.3f})"
        for f in shap_factors
    ])

    # Determine risk level
    prob = transaction_data['fraud_probability']
    if prob > 0.8:
        risk_level = "CRITICAL"
    elif prob > 0.6:
        risk_level = "HIGH"
    elif prob > 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    prompt = f"""
{RAG_CONTEXT}

TRANSACTION DETAILS:
- Transaction ID: {transaction_data['TransactionID']}
- Amount: ${transaction_data['TransactionAmt']:.2f}
- Product Category: {transaction_data['ProductCD']}
- Fraud Probability: {prob:.1%}
- Risk Level: {risk_level}
- Actual Label: {'CONFIRMED FRAUD' if transaction_data['isFraud'] == 1 else 'LEGITIMATE'}

TOP FRAUD RISK FACTORS (SHAP analysis):
{shap_text}

Write a fraud analyst report with exactly 3 sections:
1. RISK ASSESSMENT (2 sentences): Overall risk level and the main pattern detected
2. KEY EVIDENCE (2 sentences): The specific data points that drive this flag
3. RECOMMENDED ACTION (1 sentence): What the fraud team should do right now

Be specific — use the actual transaction values and SHAP impacts.
Sound like a senior fraud analyst writing for a compliance team.
Keep total response under 120 words.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

print("LLM explainer function built")

print("\n" + "=" * 60)
print("STEP 3 — Generating reports for high-risk transactions")
print("=" * 60)

# Get top 10 highest-risk transactions from sample
high_risk_sample = shap_values[
    shap_values['fraud_probability'] > 0.7
].sort_values('fraud_probability', ascending=False).head(10)

print(f"Generating reports for {len(high_risk_sample)} "
      f"high-risk transactions...")

fraud_reports = []

for _, trans in high_risk_sample.iterrows():
    trans_id = trans['TransactionID']
    shap_factors = get_shap_factors(trans_id)

    print(f"  Analyzing {trans_id} "
          f"(${trans['TransactionAmt']:.2f}, "
          f"{trans['fraud_probability']:.1%} risk)...")

    report = generate_fraud_report(trans, shap_factors)

    fraud_reports.append({
        'TransactionID': trans_id,
        'TransactionAmt': trans['TransactionAmt'],
        'ProductCD': trans['ProductCD'],
        'isFraud': trans['isFraud'],
        'fraud_probability': trans['fraud_probability'],
        'llm_report': report
    })

    time.sleep(0.5)

print("\n" + "=" * 60)
print("STEP 4 — Sample fraud analyst reports")
print("=" * 60)

for report in fraud_reports[:3]:
    print(f"\n{'='*50}")
    print(f"Transaction: {report['TransactionID']}")
    print(f"Amount: ${report['TransactionAmt']:.2f} | "
          f"Product: {report['ProductCD']} | "
          f"Risk: {report['fraud_probability']:.1%} | "
          f"Actual: {'FRAUD' if report['isFraud']==1 else 'LEGIT'}")
    print(f"\n{report['llm_report']}")

print("\n" + "=" * 60)
print("STEP 5 — Saving reports")
print("=" * 60)

reports_df = pd.DataFrame(fraud_reports)
reports_df.to_csv('outputs/fraud_reports.csv', index=False)

print(f"Saved: outputs/fraud_reports.csv")
print(f"Reports generated: {len(reports_df)}")
print(f"Confirmed fraud in reports: {reports_df['isFraud'].sum()}")
print(f"Precision: {reports_df['isFraud'].mean()*100:.1f}%")

print("\n" + "=" * 60)
print("STEP 6 — Business impact summary")
print("=" * 60)

full_high_risk = predictions[predictions['risk_level'] == 'High']

print(f"\nFull dataset fraud summary:")
print(f"High risk transactions: {len(full_high_risk):,}")
print(f"Confirmed fraud in high risk: {full_high_risk['isFraud'].sum():,}")
print(f"Precision in high risk: {full_high_risk['isFraud'].mean()*100:.1f}%")
print(f"Total amount protected: ${full_high_risk['TransactionAmt'].sum():,.0f}")

print(f"\nLLM explainability added to top {len(reports_df)} transactions")
print(f"Analyst review time saved: ~15 mins per transaction")
print(f"Total analyst time saved: ~{len(reports_df)*15} minutes")
print("Outputs:")
print("  outputs/fraud_reports.csv")