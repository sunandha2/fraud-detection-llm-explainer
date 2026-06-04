import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Fraud Detection Analyst",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; }
    .main { background-color: #0a0a0f; }
    .metric-card {
        background: #111118;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #1e1e2e;
        text-align: center;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: #4fc3f7; }
    .metric-label { font-size: 13px; color: #666; margin-top: 4px; }
    .critical { background: #2d0a0a; border-left: 4px solid #e74c3c;
                border-radius: 6px; padding: 12px 16px; margin: 8px 0; }
    .safe { background: #0a2d0a; border-left: 4px solid #2ecc71;
            border-radius: 6px; padding: 12px 16px; margin: 8px 0; }
    .report-box { background: #111118; border-left: 4px solid #4fc3f7;
                  border-radius: 6px; padding: 16px; margin: 12px 0;
                  font-size: 14px; line-height: 1.7; color: #ddd; }
    .pattern-badge { font-size: 16px; font-weight: bold; margin-bottom: 8px; }
    h1, h2, h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    predictions = pd.read_csv('outputs/fraud_predictions.csv')
    reports = pd.read_csv('outputs/fraud_reports.csv')
    shap_values = pd.read_csv('outputs/shap_values.csv')
    feature_cols = joblib.load('models/feature_cols.pkl')
    return predictions, reports, shap_values, feature_cols

@st.cache_resource
def load_model():
    return joblib.load('models/fraud_model.pkl')

predictions, reports, shap_values, feature_cols = load_data()
model = load_model()

# ── PATTERN COLORS (used in Page 3) ───────────────────────────
PATTERN_COLORS = {
    'VELOCITY_FRAUD':       '#e74c3c',
    'ORGANIZED_FRAUD_RING': '#e74c3c',
    'CARD_TESTING':         '#f39c12',
    'ACCOUNT_TAKEOVER':     '#f39c12',
    'FRIENDLY_FRAUD':       '#2ecc71',
    'UNKNOWN_PATTERN':      '#888888',
}

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Fraud Detection")
    st.markdown("*XGBoost + SHAP + Groq LLM*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Transaction Explorer", "AI Analyst Report", "Model Insights"],
        label_visibility="collapsed",
        key="nav"
    )

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"- {len(predictions):,} transactions")
    st.markdown(f"- {predictions['isFraud'].sum():,} confirmed fraud")
    st.markdown(f"- {predictions['isFraud'].mean()*100:.1f}% fraud rate")
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.markdown("- ROC-AUC: 0.90")
    st.markdown("- High risk precision: 88.2%")
    st.markdown("- Amount protected: $846,978")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("XGBoost · SHAP · Groq · Streamlit")

# ── PAGE 1: OVERVIEW ───────────────────────────────────────────
if "Overview" in page:
    st.markdown("# Fraud Detection Dashboard")
    st.markdown("*590,540 real financial transactions — IEEE-CIS Dataset*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    high_risk = predictions[predictions['risk_level'] == 'High']
    medium_risk = predictions[predictions['risk_level'] == 'Medium']

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">
                {len(high_risk):,}
            </div>
            <div class="metric-label">High Risk Transactions</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#f39c12">
                {len(medium_risk):,}
            </div>
            <div class="metric-label">Medium Risk</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#e74c3c">
                88.2%
            </div>
            <div class="metric-label">Precision (High Risk)</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#2ecc71">
                $846,978
            </div>
            <div class="metric-label">Amount Protected</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = predictions['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        fig1 = px.pie(
            risk_counts,
            values='Count',
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={
                'High': '#e74c3c',
                'Medium': '#f39c12',
                'Low': '#2ecc71'
            },
            title='Transaction Risk Distribution',
            hole=0.4
        )
        fig1.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        product_fraud = predictions.groupby('ProductCD').agg(
            total=('TransactionID', 'count'),
            fraud=('isFraud', 'sum')
        ).reset_index()
        product_fraud['fraud_rate'] = (
            product_fraud['fraud'] /
            product_fraud['total'] * 100
        ).round(1)

        fig2 = px.bar(
            product_fraud.sort_values('fraud_rate', ascending=False),
            x='ProductCD',
            y='fraud_rate',
            color='fraud_rate',
            color_continuous_scale=['#2ecc71', '#e74c3c'],
            title='Fraud Rate by Product Category',
            text='fraud_rate'
        )
        fig2.update_traces(texttemplate='%{text:.1f}%',
                           textposition='outside')
        fig2.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=350,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # High risk table
    st.markdown("### High Risk Transactions")
    display = high_risk[[
        'TransactionID', 'TransactionAmt',
        'ProductCD', 'isFraud', 'fraud_probability'
    ]].head(20).copy()
    display['fraud_probability'] = display[
        'fraud_probability'
    ].apply(lambda x: f"{x:.1%}")
    display['isFraud'] = display['isFraud'].apply(
        lambda x: "FRAUD" if x == 1 else "Legitimate"
    )
    display.columns = [
        'Transaction ID', 'Amount', 'Product',
        'Actual', 'Fraud Probability'
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ── PAGE 2: TRANSACTION EXPLORER ──────────────────────────────
elif "Explorer" in page:
    st.markdown("# Transaction Explorer")
    st.markdown("*Search any transaction and see its fraud analysis*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "High", "Medium", "Low"]
        )
    with col2:
        product_filter = st.selectbox(
            "Product",
            ["All"] + sorted(predictions['ProductCD'].unique().tolist())
        )
    with col3:
        fraud_filter = st.selectbox(
            "Actual Label",
            ["All", "Fraud Only", "Legitimate Only"]
        )

    filtered = predictions.copy()
    if risk_filter != "All":
        filtered = filtered[filtered['risk_level'] == risk_filter]
    if product_filter != "All":
        filtered = filtered[filtered['ProductCD'] == product_filter]
    if fraud_filter == "Fraud Only":
        filtered = filtered[filtered['isFraud'] == 1]
    elif fraud_filter == "Legitimate Only":
        filtered = filtered[filtered['isFraud'] == 0]

    st.markdown(f"**{len(filtered):,} transactions matching filters**")

    display = filtered[[
        'TransactionID', 'TransactionAmt',
        'ProductCD', 'isFraud', 'fraud_probability', 'risk_level'
    ]].head(50).copy()
    display['fraud_probability'] = display[
        'fraud_probability'
    ].apply(lambda x: f"{x:.1%}")
    display['isFraud'] = display['isFraud'].apply(
        lambda x: "FRAUD" if x == 1 else "Legitimate"
    )
    display.columns = [
        'Transaction ID', 'Amount ($)', 'Product',
        'Actual', 'Fraud Probability', 'Risk Level'
    ]
    st.dataframe(display, use_container_width=True, hide_index=True)

# ── PAGE 3: AI ANALYST REPORT ─────────────────────────────────
elif "AI Analyst" in page:
    st.markdown("# AI Analyst Reports")
    st.markdown("*Groq LLM classifies fraud patterns from SHAP value combinations*")
    st.markdown("---")

    # ── PRE-GENERATED REPORTS ─────────────────────────────────
    st.markdown("### Pre-Generated Reports — Top 10 High Risk")

    for _, report in reports.iterrows():
        actual = "CONFIRMED FRAUD" if report['isFraud'] == 1 else "LEGITIMATE"
        pattern = report.get('pattern', 'UNKNOWN_PATTERN')
        pcolor = PATTERN_COLORS.get(pattern, '#4fc3f7')

        with st.expander(
            f"Transaction {report['TransactionID']} — "
            f"${report['TransactionAmt']:.2f} | "
            f"Product {report['ProductCD']} | "
            f"{report['fraud_probability']:.1%} risk | "
            f"{actual}"
        ):
            # Pattern badge
            st.markdown(
                f"<div class='pattern-badge' style='color:{pcolor}'>"
                f"⚠️ {pattern.replace('_', ' ')}</div>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Confidence:** {report.get('confidence', 'N/A')}")
            with col2:
                st.markdown(f"**Action:** {report.get('action', 'N/A')}")

            st.markdown(f"""
            <div class="report-box">
                {report['llm_report']}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── LIVE REPORT GENERATOR ─────────────────────────────────
    st.markdown("### Generate Fresh Report")
    st.markdown("*Enter a Transaction ID to generate a live pattern classification*")

    trans_input = st.text_input(
        "Transaction ID",
        placeholder="e.g. 3405227"
    )

    if st.button("Generate Report"):
        if trans_input:
            try:
                trans_id = int(trans_input)
                trans_data = shap_values[
                    shap_values['TransactionID'] == trans_id
                ]

                if trans_data.empty:
                    st.error("Transaction ID not found in sample dataset")
                else:
                    with st.spinner("Groq LLM classifying fraud pattern..."):
                        trans_row = trans_data.iloc[0]

                        # Build SHAP factors
                        shap_cols = [c for c in shap_values.columns
                                     if c.startswith('shap_')]
                        factors = []
                        for col in shap_cols:
                            feature_name = col.replace('shap_', '')
                            shap_val = float(trans_row[col])
                            factors.append({
                                'feature': feature_name,
                                'shap_value': shap_val
                            })
                        factors.sort(
                            key=lambda x: abs(x['shap_value']),
                            reverse=True
                        )
                        top_factors = factors[:5]

                        shap_text = "\n".join([
                            f"- {f['feature']}: "
                            f"{'INCREASES' if f['shap_value'] > 0 else 'decreases'} "
                            f"fraud risk by {abs(f['shap_value']):.3f}"
                            for f in top_factors
                        ])

                        # Extract key signals for pattern reasoning
                        c1_impact = next((abs(f['shap_value']) for f in top_factors
                                          if f['feature'] == 'C1'), 0)
                        c8_impact = next((abs(f['shap_value']) for f in top_factors
                                          if f['feature'] == 'C8'), 0)
                        is_night = int(trans_row.get('is_night', 0))
                        product = trans_row.get('ProductCD', 'Unknown')
                        amount = float(trans_row['TransactionAmt'])

                        # Get API key — Streamlit secrets first, .env fallback
                        groq_key = (
                            st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
                            if hasattr(st, 'secrets')
                            else os.getenv("GROQ_API_KEY")
                        )

                        groq_client = Groq(api_key=groq_key)

                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a senior fraud analyst. Always respond "
                                        "in the exact 5-line format requested: PATTERN, "
                                        "EVIDENCE, CONFIDENCE, ACTION, ANALYST_NOTE. "
                                        "Never add preamble or extra text before PATTERN:."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": f"""
You are a senior fraud analyst. Classify the fraud pattern from SHAP value combinations.

TRANSACTION:
- ID: {trans_id}
- Amount: ${amount:.2f}
- Product: {product} ({'HIGHEST RISK' if product == 'C' else 'standard risk'})
- Fraud Probability: {trans_row['fraud_probability']:.1%}
- Time: {'NIGHT' if is_night == 1 else 'DAY'}
- Actual: {'CONFIRMED FRAUD' if trans_row['isFraud'] == 1 else 'LEGITIMATE'}

TOP SHAP DRIVERS:
{shap_text}

Key signal strengths: C1 impact={c1_impact:.3f}, C8 impact={c8_impact:.3f}, Amount=${amount:.2f}

CLASSIFY as exactly ONE pattern:
- VELOCITY_FRAUD: C1 or C8 SHAP impact dominates (>0.3) — stolen card being drained
- CARD_TESTING: amount under $100 AND high C1/C8 impact — testing stolen card validity
- ORGANIZED_FRAUD_RING: night + Product C + amount over $200 all present together
- ACCOUNT_TAKEOVER: email_match or addr features dominate SHAP — identity mismatch
- FRIENDLY_FRAUD: no single SHAP feature dominates, borderline probability
- UNKNOWN_PATTERN: signals contradict each other

CONFIDENCE: HIGH (3+ signals align) / MEDIUM (2 align) / LOW (mixed)

ACTION rules:
- VELOCITY_FRAUD + amount>$200: Block card immediately, call cardholder within 1 hour
- VELOCITY_FRAUD + amount<=$200: Block card silently, flag for 24-hour monitoring
- CARD_TESTING: Block card silently, flag account for 24-hour monitoring
- ORGANIZED_FRAUD_RING: Block immediately + escalate to fraud investigation team
- ACCOUNT_TAKEOVER: Do not block — flag for identity verification within 2 hours
- FRIENDLY_FRAUD or LOW confidence: Flag for manual review within 4 hours, do not auto-block

RESPOND IN EXACTLY THIS FORMAT:
PATTERN: [pattern name]
EVIDENCE: [2-3 sentences on what the COMBINATION of signals means]
CONFIDENCE: [HIGH/MEDIUM/LOW] — [one sentence why]
ACTION: [exact action]
ANALYST_NOTE: [one insight not obvious from the raw numbers]"""
                                }
                            ],
                            max_tokens=350,
                            temperature=0.1
                        )

                        fresh_report = response.choices[0].message.content.strip()

                        # Parse structured fields
                        lines = fresh_report.split('\n')
                        pattern = next(
                            (l.replace('PATTERN:', '').strip()
                             for l in lines if l.startswith('PATTERN:')),
                            'UNKNOWN_PATTERN'
                        )
                        confidence = next(
                            (l.replace('CONFIDENCE:', '').strip()
                             for l in lines if l.startswith('CONFIDENCE:')),
                            'N/A'
                        )
                        action = next(
                            (l.replace('ACTION:', '').strip()
                             for l in lines if l.startswith('ACTION:')),
                            'N/A'
                        )

                        pcolor = PATTERN_COLORS.get(pattern, '#4fc3f7')

                        # Display
                        st.markdown(
                            f"<h3 style='color:{pcolor}'>⚠️ {pattern.replace('_', ' ')}</h3>",
                            unsafe_allow_html=True
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Confidence:** {confidence}")
                        with col2:
                            st.markdown(f"**Action:** {action}")

                        st.markdown(f"""
                        <div class="report-box">
                            {fresh_report}
                        </div>""", unsafe_allow_html=True)

            except ValueError:
                st.error("Please enter a valid Transaction ID number")

# ── PAGE 4: MODEL INSIGHTS ─────────────────────────────────────
elif "Insights" in page:
    st.markdown("# Model Insights")
    st.markdown("*How the XGBoost model makes decisions*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        shap_cols = [c for c in shap_values.columns if c.startswith('shap_')]
        mean_shap = shap_values[shap_cols].abs().mean()
        importance_df = pd.DataFrame({
            'feature': [c.replace('shap_', '') for c in shap_cols],
            'importance': mean_shap.values
        }).sort_values('importance', ascending=True).tail(15)

        fig3 = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            color='importance',
            color_continuous_scale=['#3498db', '#e74c3c'],
            title='Top Features — SHAP Importance'
        )
        fig3.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=450,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.histogram(
            predictions,
            x='fraud_probability',
            color='risk_level',
            color_discrete_map={
                'High': '#e74c3c',
                'Medium': '#f39c12',
                'Low': '#2ecc71'
            },
            title='Fraud Probability Distribution',
            nbins=50
        )
        fig4.update_layout(
            plot_bgcolor='#111118',
            paper_bgcolor='#0a0a0f',
            font=dict(color='white'),
            height=450
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("ROC-AUC", "0.90", "#4fc3f7"),
        ("Precision (High Risk)", "88.2%", "#2ecc71"),
        ("Recall (Fraud)", "35%", "#f39c12"),
        ("Features Used", "45", "#4fc3f7"),
    ]

    for col, (label, value, color) in zip(
        [col1, col2, col3, col4], metrics
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">
                    {value}
                </div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)