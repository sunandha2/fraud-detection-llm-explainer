import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings('ignore')

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
load_dotenv()


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

FRAUD PATTERN DEFINITIONS:
- VELOCITY_FRAUD: Card used abnormally many times in short period.
  Signal: C1 or C8 SHAP impact dominates (> 0.5). Stolen card being drained.
- CARD_TESTING: Fraudster tests if stolen card is valid with small amounts.
  Signal: Low TransactionAmt (under $100) + high C1/C8 SHAP impact.
- ORGANIZED_FRAUD_RING: Coordinated attack, multiple signals align.
  Signal: Night transaction + Product C + high amount all present together.
- ACCOUNT_TAKEOVER: Stolen account credentials, not just card.
  Signal: email_match or addr SHAP features dominate — identity mismatch.
- FRIENDLY_FRAUD: Cardholder disputes legitimate transaction.
  Signal: Mixed low-impact SHAP values, no single dominant signal, borderline probability.
- UNKNOWN_PATTERN: Signals contradict each other, cannot classify.
"""

print("\n" + "=" * 60)
print("STEP 2 — Building explanation functions")
print("=" * 60)


def get_shap_factors(transaction_id, top_n=5):
    """Get top SHAP factors for a transaction"""
    row = shap_values[shap_values['TransactionID'] == transaction_id]
    if row.empty:
        return []

    row = row.iloc[0]
    shap_cols = [c for c in shap_values.columns if c.startswith('shap_')]

    factors = []
    for col in shap_cols:
        feature_name = col.replace('shap_', '')
        shap_val = row[col]
        factors.append({
            'feature': feature_name,
            'shap_value': float(shap_val),
            'direction': 'INCREASES' if shap_val > 0 else 'decreases'
        })

    factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return factors[:top_n]


def parse_llm_response(response_text: str) -> dict:
    """
    Parse the structured LLM response into a dict.
    Makes output usable programmatically in Streamlit UI.
    """
    result = {
        "pattern": "UNKNOWN_PATTERN",
        "evidence": "",
        "confidence": "LOW",
        "confidence_reason": "",
        "action": "Flag for manual review within 4 hours",
        "analyst_note": ""
    }

    lines = response_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("PATTERN:"):
            result["pattern"] = line.replace("PATTERN:", "").strip()
        elif line.startswith("EVIDENCE:"):
            result["evidence"] = line.replace("EVIDENCE:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            conf_text = line.replace("CONFIDENCE:", "").strip()
            for level in ["HIGH", "MEDIUM", "LOW"]:
                if level in conf_text.upper():
                    result["confidence"] = level
                    break
            # Capture the reason after the dash if present
            if "—" in conf_text:
                result["confidence_reason"] = conf_text.split("—", 1)[1].strip()
            elif "-" in conf_text:
                result["confidence_reason"] = conf_text.split("-", 1)[1].strip()
        elif line.startswith("ACTION:"):
            result["action"] = line.replace("ACTION:", "").strip()
        elif line.startswith("ANALYST_NOTE:"):
            result["analyst_note"] = line.replace("ANALYST_NOTE:", "").strip()

    return result


def generate_fraud_report(transaction_data, shap_factors):
    """
    Uses LLM to CLASSIFY fraud pattern type from SHAP value combinations.

    This is what separates LLM from an f-string template:
    It reasons about what C1=108 AND C8=100 AND is_night=1 TOGETHER mean,
    classifies a pattern type from 6 options, rates its own confidence
    based on how many signals align, and adjusts its action recommendation
    conditionally based on pattern + amount. An f-string cannot do this.
    """

    # Format SHAP factors for the prompt
    shap_text = "\n".join([
        f"  - {f['feature']}: {f['direction']} fraud risk "
        f"(SHAP impact: {abs(f['shap_value']):.3f})"
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

    # Extract key signal values for the prompt context
    c1_impact = next((abs(f['shap_value']) for f in shap_factors if f['feature'] == 'C1'), 0)
    c8_impact = next((abs(f['shap_value']) for f in shap_factors if f['feature'] == 'C8'), 0)
    is_night = transaction_data.get('is_night', 0)
    product = transaction_data.get('ProductCD', 'Unknown')
    amount = transaction_data.get('TransactionAmt', 0)

    prompt = f"""
{RAG_CONTEXT}

TRANSACTION DETAILS:
- Transaction ID: {transaction_data['TransactionID']}
- Amount: ${amount:.2f}
- Product Category: {product}
- Fraud Probability: {prob:.1%}
- Risk Level: {risk_level}
- Time of day: {'NIGHT (10pm-6am)' if is_night == 1 else 'DAY'}
- Actual Label: {'CONFIRMED FRAUD' if transaction_data['isFraud'] == 1 else 'LEGITIMATE'}

TOP SHAP DRIVERS — these are the features that pushed the model to this decision:
{shap_text}

Extracted signal strengths for your reasoning:
- C1 SHAP impact: {c1_impact:.3f}
- C8 SHAP impact: {c8_impact:.3f}
- Night transaction: {'YES' if is_night == 1 else 'NO'}
- Product: {product} ({'HIGHEST RISK category' if product == 'C' else 'standard risk category'})
- Amount: ${amount:.2f} ({'high value' if amount > 200 else 'low value'})

=== YOUR TASK ===

STEP 1 — CLASSIFY the fraud pattern. Pick EXACTLY ONE from this list:
- VELOCITY_FRAUD: if C1 or C8 SHAP impact is the top driver (> 0.3) — card used abnormally many times
- CARD_TESTING: if amount is under $100 AND C1/C8 SHAP impact is high — testing stolen card validity
- ORGANIZED_FRAUD_RING: if night=YES AND product=C AND amount > $200 all occur together
- ACCOUNT_TAKEOVER: if email_match or addr features appear in top SHAP drivers
- FRIENDLY_FRAUD: if no single SHAP feature dominates (all impacts under 0.3) AND probability is borderline (30-70%)
- UNKNOWN_PATTERN: only if signals strongly contradict each other

STEP 2 — Write EVIDENCE explaining the COMBINATION of signals.
Do NOT just list features. Explain what they mean together.
GOOD: "C1 and C8 both ranking as top SHAP drivers simultaneously means the card
is being used across many transactions rapidly — this is the fingerprint of a
stolen card that has been sold and is being drained before cancellation."
BAD: "C1 is high and increases fraud risk."

STEP 3 — Rate your CONFIDENCE:
- HIGH: 3 or more SHAP drivers align with your chosen pattern
- MEDIUM: 2 drivers align, others are neutral
- LOW: signals are mixed or only 1 driver aligns

STEP 4 — Recommend ONE specific ACTION:
- VELOCITY_FRAUD + amount > $200: Block card immediately, call cardholder within 1 hour
- VELOCITY_FRAUD + amount <= $200: Block card silently, flag for 24-hour monitoring
- CARD_TESTING: Block card silently, flag account for 24-hour monitoring
- ORGANIZED_FRAUD_RING: Block immediately + escalate to fraud investigation team
- ACCOUNT_TAKEOVER: Do not block — flag for identity verification call within 2 hours
- FRIENDLY_FRAUD or UNKNOWN_PATTERN or LOW confidence: Flag for manual review within 4 hours, do not auto-block

STEP 5 — Write ONE ANALYST_NOTE with an insight not obvious from the raw numbers.
Think about what this pattern means operationally, what the fraudster's likely intent is,
or what follow-up the team should do beyond the immediate action.

=== RESPOND IN EXACTLY THIS FORMAT — NO PREAMBLE, NO EXTRA TEXT ===
PATTERN: [exactly one pattern name from the list]
EVIDENCE: [2-3 sentences on what the COMBINATION means]
CONFIDENCE: [HIGH/MEDIUM/LOW] — [one sentence why]
ACTION: [exact recommended action]
ANALYST_NOTE: [one operational insight not obvious from the numbers]
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior fraud analyst. Always respond in the exact 5-line format "
                    "requested: PATTERN, EVIDENCE, CONFIDENCE, ACTION, ANALYST_NOTE. "
                    "Never add preamble, greetings, or any text before PATTERN:. "
                    "Never explain your reasoning outside the format."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=350,
        temperature=0.1  # Low temperature = consistent, analytical responses
    )

    return response.choices[0].message.content.strip()


print("LLM explainer function built")

print("\n" + "=" * 60)
print("STEP 3 — Generating reports for high-risk transactions")
print("=" * 60)

# Get top 10 highest-risk transactions
high_risk_sample = shap_values[
    shap_values['fraud_probability'] > 0.7
].sort_values('fraud_probability', ascending=False).head(10)

print(f"Generating reports for {len(high_risk_sample)} high-risk transactions...")

fraud_reports = []

for _, trans in high_risk_sample.iterrows():
    trans_id = trans['TransactionID']
    shap_factors = get_shap_factors(trans_id)

    print(f"  Analyzing {trans_id} "
          f"(${trans['TransactionAmt']:.2f}, "
          f"{trans['fraud_probability']:.1%} risk)...")

    report = generate_fraud_report(trans, shap_factors)

    # Parse the structured response
    parsed = parse_llm_response(report)

    fraud_reports.append({
        'TransactionID': trans_id,
        'TransactionAmt': trans['TransactionAmt'],
        'ProductCD': trans['ProductCD'],
        'isFraud': trans['isFraud'],
        'fraud_probability': trans['fraud_probability'],
        'llm_report': report,          # full raw text
        'pattern': parsed['pattern'],  # structured fields for UI
        'confidence': parsed['confidence'],
        'action': parsed['action'],
        'evidence': parsed['evidence'],
        'analyst_note': parsed['analyst_note']
    })

    time.sleep(0.5)

print("\n" + "=" * 60)
print("STEP 4 — Sample fraud analyst reports")
print("=" * 60)

for report in fraud_reports[:3]:
    print(f"\n{'=' * 55}")
    print(f"Transaction : {report['TransactionID']}")
    print(f"Amount      : ${report['TransactionAmt']:.2f}  |  "
          f"Product: {report['ProductCD']}  |  "
          f"Risk: {report['fraud_probability']:.1%}  |  "
          f"Actual: {'FRAUD' if report['isFraud'] == 1 else 'LEGIT'}")
    print(f"\n{report['llm_report']}")
    print(f"\n  >>> PATTERN    : {report['pattern']}")
    print(f"  >>> CONFIDENCE : {report['confidence']}")
    print(f"  >>> ACTION     : {report['action']}")

# ── PATTERN DISTRIBUTION ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4b — Pattern distribution across flagged transactions")
print("=" * 60)

reports_df_preview = pd.DataFrame(fraud_reports)
pattern_counts = reports_df_preview['pattern'].value_counts()
print("\nFraud patterns detected:")
for pattern, count in pattern_counts.items():
    pct = count / len(fraud_reports) * 100
    print(f"  {pattern:<25} {count:>3} transactions  ({pct:.0f}%)")

print("\n" + "=" * 60)
print("STEP 5 — Saving reports")
print("=" * 60)

reports_df = pd.DataFrame(fraud_reports)
reports_df.to_csv('outputs/fraud_reports.csv', index=False)

print(f"Saved  : outputs/fraud_reports.csv")
print(f"Columns: {list(reports_df.columns)}")
print(f"Reports generated       : {len(reports_df)}")
print(f"Confirmed fraud in reports: {reports_df['isFraud'].sum()}")
print(f"Precision               : {reports_df['isFraud'].mean() * 100:.1f}%")

print("\n" + "=" * 60)
print("STEP 6 — Business impact summary")
print("=" * 60)

full_high_risk = predictions[predictions['risk_level'] == 'High']

print(f"\nFull dataset fraud summary:")
print(f"High risk transactions      : {len(full_high_risk):,}")
print(f"Confirmed fraud in high risk: {full_high_risk['isFraud'].sum():,}")
print(f"Precision in high risk      : {full_high_risk['isFraud'].mean() * 100:.1f}%")
print(f"Total amount protected      : ${full_high_risk['TransactionAmt'].sum():,.0f}")

print(f"\nLLM pattern classification added to top {len(reports_df)} transactions")
print(f"Analyst review time saved: ~15 mins per transaction")
print(f"Total analyst time saved : ~{len(reports_df) * 15} minutes")
print("\nOutputs:")
print("  outputs/fraud_reports.csv")
print("  Columns: TransactionID, TransactionAmt, ProductCD, isFraud,")
print("           fraud_probability, llm_report, pattern,")
print("           confidence, action, evidence, analyst_note")