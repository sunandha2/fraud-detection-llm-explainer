import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(project_root):
    os.chdir(project_root)
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("STEP 1 — Loading transaction data")
print("=" * 60)

# Load only key columns first to save memory
trans = pd.read_csv('data/train_transaction.csv',
    usecols=['TransactionID', 'isFraud', 'TransactionAmt',
             'ProductCD', 'card4', 'card6', 'P_emaildomain',
             'R_emaildomain', 'TransactionDT'])

print(f"Shape: {trans.shape}")
print(f"Columns: {trans.columns.tolist()}")

print("\n" + "=" * 60)
print("STEP 2 — Fraud rate analysis")
print("=" * 60)

print(f"\nTotal transactions: {len(trans):,}")
print(f"Fraudulent: {trans['isFraud'].sum():,}")
print(f"Legitimate: {(trans['isFraud']==0).sum():,}")
print(f"Fraud rate: {trans['isFraud'].mean()*100:.2f}%")

print("\n" + "=" * 60)
print("STEP 3 — Transaction amount analysis")
print("=" * 60)

print(f"\nAll transactions:")
print(trans['TransactionAmt'].describe())

print(f"\nFraudulent transactions:")
print(trans[trans['isFraud']==1]['TransactionAmt'].describe())

print(f"\nLegitimate transactions:")
print(trans[trans['isFraud']==0]['TransactionAmt'].describe())

print("\n" + "=" * 60)
print("STEP 4 — Product category analysis")
print("=" * 60)

print(f"\nProduct categories:")
print(trans['ProductCD'].value_counts())

print(f"\nFraud rate by product:")
print(trans.groupby('ProductCD')['isFraud'].mean().sort_values(ascending=False).apply(lambda x: f"{x*100:.2f}%"))

print("\n" + "=" * 60)
print("STEP 5 — Card analysis")
print("=" * 60)

print(f"\nCard types:")
print(trans['card4'].value_counts())

print(f"\nFraud rate by card type:")
print(trans.groupby('card4')['isFraud'].mean().sort_values(ascending=False).apply(lambda x: f"{x*100:.2f}%"))

print("\n" + "=" * 60)
print("STEP 6 — Identity data")
print("=" * 60)

identity = pd.read_csv('data/train_identity.csv')
print(f"Identity data shape: {identity.shape}")
print(f"Transactions with identity: {len(identity):,}")
print(f"Identity match rate: {len(identity)/len(trans)*100:.1f}%")

print("\n" + "=" * 60)
print("STEP 7 — Missing values")
print("=" * 60)

missing = trans.isnull().sum()
missing_pct = (missing / len(trans) * 100).round(1)
missing_df = pd.DataFrame({'missing': missing, 'pct': missing_pct})
print(missing_df[missing_df['missing'] > 0])