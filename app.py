from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model at startup
model = None
feature_cols = None

def load_model():
    global model, feature_cols
    try:
        model = joblib.load('models/fraud_model.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model not found: {e}")

load_model()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #e74c3c; }
        .endpoint { background: #f4f4f4; padding: 15px; border-radius: 8px; margin: 15px 0; }
        code { background: #eee; padding: 2px 6px; border-radius: 4px; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; color: white; font-size: 12px; }
        .get { background: #27ae60; }
        .post { background: #2980b9; }
    </style>
</head>
<body>
    <h1>🔍 Fraud Detection API</h1>
    <p>XGBoost model trained on IEEE-CIS Fraud Detection dataset.</p>

    <div class="endpoint">
        <span class="badge get">GET</span> <code>/health</code>
        <p>Check if model is loaded and API is running.</p>
    </div>

    <div class="endpoint">
        <span class="badge post">POST</span> <code>/predict</code>
        <p>Submit a transaction and get a fraud probability score.</p>
        <b>Example request body:</b>
        <pre>{
  "TransactionAmt": 150.00,
  "ProductCD": "W",
  "card4": "visa",
  "card6": "debit",
  "hour": 14,
  "is_night": 0,
  "is_weekend": 0
}</pre>
    </div>

    <div class="endpoint">
        <span class="badge get">GET</span> <code>/features</code>
        <p>List all features the model uses.</p>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'features': len(feature_cols) if feature_cols else 0
    })

@app.route('/features')
def features():
    return jsonify({'features': feature_cols or []})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    # Build feature row with defaults
    row = {col: -999 for col in feature_cols}

    # Map incoming fields
    amt = float(data.get('TransactionAmt', 0))
    row['TransactionAmt'] = amt
    row['amt_log'] = np.log1p(amt)
    row['amt_rounded'] = int(amt % 1 == 0)
    row['amt_cents_99'] = int(amt % 1 > 0.98)

    row['hour'] = float(data.get('hour', 12))
    row['day_of_week'] = float(data.get('day_of_week', 3))
    row['is_night'] = int(data.get('is_night', 0))
    row['is_weekend'] = int(data.get('is_weekend', 0))

    card4_map = {'visa': 0, 'mastercard': 1, 'american express': 2, 'discover': 3}
    card6_map = {'credit': 0, 'debit': 1, 'debit or credit': 2, 'charge card': 3}
    product_map = {'W': 0, 'C': 1, 'R': 2, 'H': 3, 'S': 4}

    row['card4_enc'] = card4_map.get(str(data.get('card4', '')).lower(), -1)
    row['card6_enc'] = card6_map.get(str(data.get('card6', '')).lower(), -1)
    row['product_enc'] = product_map.get(str(data.get('ProductCD', '')).upper(), -1)

    row['p_email_isna'] = int(data.get('P_emaildomain') is None)
    row['r_email_isna'] = int(data.get('R_emaildomain') is None)
    row['email_match'] = int(
        data.get('P_emaildomain') == data.get('R_emaildomain')
    )
    risky = ['gmail.com', 'yahoo.com', 'hotmail.com', 'anonymous.com', 'protonmail.com']
    row['p_email_risky'] = int(data.get('P_emaildomain', '') in risky)

    row['has_identity'] = int(data.get('has_identity', 0))
    row['identity_count'] = int(data.get('identity_count', 0))
    row['is_mobile'] = int(data.get('is_mobile', 0))

    X = pd.DataFrame([row])[feature_cols].fillna(-999)
    prob = float(model.predict_proba(X)[0][1])

    risk = 'High' if prob > 0.6 else 'Medium' if prob > 0.3 else 'Low'

    return jsonify({
        'fraud_probability': round(prob, 4),
        'risk_level': risk,
        'is_fraud': prob > 0.5
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
