import pandas as pd
import numpy as np
import joblib
import shap
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'C:\Users\sunandha\Downloads\gitdemo\fraud-detection-llm-explainer')
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("STEP 1 — Loading model and data")
print("=" * 60)

model = joblib.load('models/fraud_model.pkl')
feature_cols = joblib.load('models/feature_cols.pkl')

# Load sample transactions for SHAP analysis
# Using sample to keep it fast — full dataset is 590K
sample = pd.read_csv('outputs/sample_transactions.csv')
predictions = pd.read_csv('outputs/fraud_predictions.csv')

print(f"Model loaded: {type(model)}")
print(f"Features: {len(feature_cols)}")
print(f"Sample transactions: {len(sample)}")

print("\n" + "=" * 60)
print("STEP 2 — Building SHAP explainer")
print("=" * 60)

# Prepare feature matrix — same as training
X_sample = sample[feature_cols].fillna(-999)

# TreeExplainer is optimized for XGBoost
explainer = shap.TreeExplainer(model)
print("SHAP TreeExplainer built")

print("Calculating SHAP values for 1,000 sample transactions...")
shap_values = explainer.shap_values(X_sample)
print(f"SHAP values shape: {shap_values.shape}")

print("\n" + "=" * 60)
print("STEP 3 — Global feature importance")
print("=" * 60)

mean_shap = np.abs(shap_values).mean(axis=0)
global_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_shap': mean_shap
}).sort_values('mean_shap', ascending=False)

print("\nTop 15 features by SHAP importance:")
print(global_importance.head(15).to_string(index=False))

global_importance.to_csv(
    'outputs/shap_global_importance.csv', index=False
)

print("\n" + "=" * 60)
print("STEP 4 — Per-transaction SHAP explanations")
print("=" * 60)

def get_transaction_explanation(trans_idx, top_n=5):
    """Get top SHAP factors for a specific transaction"""
    shap_row = shap_values[trans_idx]
    trans_features = X_sample.iloc[trans_idx]

    explanation = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_row,
        'feature_value': trans_features.values
    })

    explanation['abs_shap'] = explanation['shap_value'].abs()
    explanation = explanation.sort_values(
        'abs_shap', ascending=False
    )

    return explanation.head(top_n)

def format_fraud_explanation(trans_id, explanation_df,
                              fraud_prob, is_fraud,
                              amount, product):
    """Format SHAP explanation into readable analyst report"""

    risk = "HIGH" if fraud_prob > 0.6 else \
           "MEDIUM" if fraud_prob > 0.3 else "LOW"
    actual = "FRAUD" if is_fraud == 1 else "LEGITIMATE"

    lines = []
    lines.append(f"Transaction ID: {trans_id}")
    lines.append(f"Amount: ${amount:.2f}")
    lines.append(f"Product: {product}")
    lines.append(f"Fraud Probability: {fraud_prob:.1%}")
    lines.append(f"Risk Level: {risk}")
    lines.append(f"Actual Label: {actual}")
    lines.append(f"Top factors:")

    for _, row in explanation_df.iterrows():
        direction = "INCREASES" if row['shap_value'] > 0 \
                    else "decreases"
        impact = abs(row['shap_value'])
        lines.append(
            f"  • {row['feature']} = {row['feature_value']:.2f}"
            f" → {direction} fraud risk by {impact:.3f}"
        )

    return "\n".join(lines)

# Show explanations for high-risk flagged transactions
sample['fraud_probability'] = model.predict_proba(
    X_sample
)[:, 1]

high_risk_sample = sample[
    sample['fraud_probability'] > 0.6
].head(5)

print("\nSample explanations for high-risk transactions:")
for i, (idx, row) in enumerate(high_risk_sample.iterrows()):
    sample_idx = sample.index.get_loc(idx)
    explanation = get_transaction_explanation(sample_idx)

    text = format_fraud_explanation(
        row['TransactionID'],
        explanation,
        row['fraud_probability'],
        row['isFraud'],
        row['TransactionAmt'],
        row['ProductCD']
    )
    print(f"\n{text}")
    print("-" * 50)

print("\n" + "=" * 60)
print("STEP 5 — Fraud pattern insights")
print("=" * 60)

fraud_sample = sample[sample['isFraud'] == 1]
legit_sample = sample[sample['isFraud'] == 0]

print(f"\nSample breakdown:")
print(f"Fraudulent: {len(fraud_sample)}")
print(f"Legitimate: {len(legit_sample)}")

print(f"\nAvg transaction amount:")
print(f"Fraud: ${fraud_sample['TransactionAmt'].mean():.2f}")
print(f"Legit: ${legit_sample['TransactionAmt'].mean():.2f}")

print(f"\nNight transactions (10pm-6am):")
if 'is_night' in sample.columns:
    print(f"Fraud: {fraud_sample['is_night'].mean()*100:.1f}%")
    print(f"Legit: {legit_sample['is_night'].mean()*100:.1f}%")

print(f"\nEmail mismatch rate:")
if 'email_match' in sample.columns:
    print(f"Fraud: {(1-fraud_sample['email_match'].mean())*100:.1f}% mismatch")
    print(f"Legit: {(1-legit_sample['email_match'].mean())*100:.1f}% mismatch")

print("\n" + "=" * 60)
print("STEP 6 — Save SHAP values")
print("=" * 60)

shap_df = pd.DataFrame(
    shap_values,
    columns=[f'shap_{c}' for c in feature_cols]
)
shap_df['TransactionID'] = sample['TransactionID'].values
shap_df['isFraud'] = sample['isFraud'].values
shap_df['fraud_probability'] = sample['fraud_probability'].values
shap_df['TransactionAmt'] = sample['TransactionAmt'].values
shap_df['ProductCD'] = sample['ProductCD'].values

shap_df.to_csv('outputs/shap_values.csv', index=False)
print(f"Saved: outputs/shap_values.csv ({len(shap_df)} transactions)")

print("\n" + "=" * 60)
print("STEP 7 — Business impact summary")
print("=" * 60)

full_predictions = pd.read_csv('outputs/fraud_predictions.csv')
high_risk = full_predictions[full_predictions['risk_level'] == 'High']
medium_risk = full_predictions[full_predictions['risk_level'] == 'Medium']

print(f"\nFull dataset risk summary:")
print(f"High risk transactions:   {len(high_risk):,}")
print(f"Medium risk transactions: {len(medium_risk):,}")
print(f"\nAmount in high risk:   ${high_risk['TransactionAmt'].sum():,.0f}")
print(f"Amount in medium risk: ${medium_risk['TransactionAmt'].sum():,.0f}")
print(f"\nActual fraud in high risk bucket: {high_risk['isFraud'].sum():,}")
print(f"Precision in high risk: {high_risk['isFraud'].mean()*100:.1f}%")

print(f"\nTop 5 global fraud drivers (SHAP):")
for _, row in global_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['mean_shap']:.4f} avg impact")

print("Outputs:")
print("  outputs/shap_global_importance.csv")
print("  outputs/shap_values.csv")