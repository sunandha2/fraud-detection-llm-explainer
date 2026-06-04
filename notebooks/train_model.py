import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(project_root):
    os.chdir(project_root)
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

trans = pd.read_csv('data/train_transaction.csv')
identity = pd.read_csv('data/train_identity.csv')

print(f"Transactions: {trans.shape}")
print(f"Identity: {identity.shape}")

# Merge — left join keeps all transactions
df = trans.merge(identity, on='TransactionID', how='left')
print(f"Merged shape: {df.shape}")
print(f"Fraud rate: {df['isFraud'].mean()*100:.2f}%")

print("\n" + "=" * 60)
print("STEP 2 — Feature Engineering")
print("=" * 60)

# ── TIME FEATURES ──
# TransactionDT is seconds from a reference point
df['hour'] = (df['TransactionDT'] / 3600) % 24
df['day_of_week'] = (df['TransactionDT'] / (3600 * 24)) % 7
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
print("Time features: hour, day_of_week, is_night, is_weekend")

# ── AMOUNT FEATURES ──
df['amt_log'] = np.log1p(df['TransactionAmt'])
df['amt_rounded'] = (df['TransactionAmt'] % 1 == 0).astype(int)
df['amt_cents_99'] = (df['TransactionAmt'] % 1 > 0.98).astype(int)
print(" Amount features: log amount, rounded flag, .99 cents flag")

# ── EMAIL FEATURES ──
df['p_email_isna'] = df['P_emaildomain'].isna().astype(int)
df['r_email_isna'] = df['R_emaildomain'].isna().astype(int)
df['email_match'] = (
    df['P_emaildomain'] == df['R_emaildomain']
).astype(int)

# High risk email domains
risky_domains = ['gmail.com', 'yahoo.com', 'hotmail.com',
                 'anonymous.com', 'protonmail.com']
df['p_email_risky'] = df['P_emaildomain'].isin(risky_domains).astype(int)
print("Email features: missing flags, domain match, risky domains")

# ── CARD FEATURES ──
# Encode card type
card4_map = {'visa': 0, 'mastercard': 1,
             'american express': 2, 'discover': 3}
df['card4_enc'] = df['card4'].map(card4_map).fillna(-1)

card6_map = {'credit': 0, 'debit': 1,
             'debit or credit': 2, 'charge card': 3}
df['card6_enc'] = df['card6'].map(card6_map).fillna(-1)
print(" Card features: card type encoded")

# ── PRODUCT FEATURES ──
product_map = {'W': 0, 'C': 1, 'R': 2, 'H': 3, 'S': 4}
df['product_enc'] = df['ProductCD'].map(product_map).fillna(-1)
print("Product features: product category encoded")

# ── IDENTITY FEATURES ──
# DeviceType — mobile vs desktop
df['is_mobile'] = (df.get('DeviceType', pd.Series(
    ['unknown'] * len(df))) == 'mobile').astype(int)

# How many identity fields are filled
id_cols = [c for c in df.columns if c.startswith('id_')]
if id_cols:
    df['identity_count'] = df[id_cols].notna().sum(axis=1)
    df['has_identity'] = (df['identity_count'] > 0).astype(int)
else:
    df['identity_count'] = 0
    df['has_identity'] = 0
print("Identity features: device type, identity completeness")

print("\n" + "=" * 60)
print("STEP 3 — Selecting features")
print("=" * 60)

FEATURE_COLS = [
    # Amount
    'TransactionAmt', 'amt_log', 'amt_rounded', 'amt_cents_99',
    # Time
    'hour', 'day_of_week', 'is_night', 'is_weekend',
    # Card
    'card4_enc', 'card6_enc',
    'card1', 'card2', 'card3', 'card5',
    # Product
    'product_enc',
    # Email
    'p_email_isna', 'r_email_isna',
    'email_match', 'p_email_risky',
    # Identity
    'has_identity', 'identity_count', 'is_mobile',
    # C fields (transaction count aggregates)
    'C1', 'C2', 'C3', 'C4', 'C5',
    'C6', 'C7', 'C8', 'C9', 'C10',
    # D fields (timedelta features)
    'D1', 'D2', 'D3', 'D4',
    # V fields (Vesta engineered features — top ones)
    'V1', 'V2', 'V3', 'V4', 'V5',
    'V12', 'V13', 'V14', 'V15',
]

# Keep only columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"Features selected: {len(FEATURE_COLS)}")
print(f"Feature list: {FEATURE_COLS}")

X = df[FEATURE_COLS].fillna(-999)
y = df['isFraud'].astype(int)

print(f"\nX shape: {X.shape}")
print(f"Fraud rate: {y.mean()*100:.2f}%")

print("\n" + "=" * 60)
print("STEP 4 — Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n" + "=" * 60)
print("STEP 5 — SMOTE for class imbalance")
print("=" * 60)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_bal).value_counts().to_dict()}")

print("\n" + "=" * 60)
print("STEP 6 — Training XGBoost")
print("=" * 60)
print("This takes 3-5 minutes...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='auc',
    verbosity=0,
    tree_method='hist'
)

model.fit(
    X_train_bal, y_train_bal,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("Model trained!")

print("\n" + "=" * 60)
print("STEP 7 — Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Fraud']))

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.4f}")

print("\n" + "=" * 60)
print("STEP 8 — Feature Importance")
print("=" * 60)

importance_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 most important features:")
print(importance_df.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("STEP 9 — Generate fraud predictions")
print("=" * 60)

# Score ALL transactions
df['fraud_probability'] = model.predict_proba(
    df[FEATURE_COLS].fillna(-999)
)[:, 1]

df['risk_level'] = pd.cut(
    df['fraud_probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

risk_summary = df.groupby('risk_level', observed=True).agg(
    transactions=('TransactionID', 'count'),
    fraud_count=('isFraud', 'sum'),
    avg_amount=('TransactionAmt', 'mean'),
    total_amount=('TransactionAmt', 'sum')
).reset_index()

print("\nRisk Distribution:")
print(risk_summary.to_string(index=False))

high_risk = df[df['risk_level'] == 'High']
print(f"\nHigh risk transactions: {len(high_risk):,}")
print(f"Amount at risk: ${high_risk['TransactionAmt'].sum():,.0f}")
print(f"Actual fraud in high risk: {high_risk['isFraud'].sum():,}")
print(f"Precision in high risk bucket: {high_risk['isFraud'].mean()*100:.1f}%")

print("\n" + "=" * 60)
print("STEP 10 — Saving")
print("=" * 60)

joblib.dump(model, 'models/fraud_model.pkl')
joblib.dump(FEATURE_COLS, 'models/feature_cols.pkl')

# Save sample for Day 3 SHAP analysis
sample = df.sample(n=1000, random_state=42)
sample.to_csv('outputs/sample_transactions.csv', index=False)

# Save full predictions
df[['TransactionID', 'isFraud', 'TransactionAmt',
    'ProductCD', 'fraud_probability',
    'risk_level']].to_csv(
    'outputs/fraud_predictions.csv', index=False)

print("Saved: models/fraud_model.pkl")
print("Saved: models/feature_cols.pkl")
print("Saved: outputs/sample_transactions.csv")
print("Saved: outputs/fraud_predictions.csv")
print(f"\nROC-AUC: {auc:.4f}")
