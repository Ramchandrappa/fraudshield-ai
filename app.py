"""
FraudShield AI - AI-Powered Fraud Detection System (Streamlit web app)

Run with:   streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import CATEGORIES, CHANNELS
from fraud_model import (DATA_PATH, METRICS_PATH, MODEL_PATH, FraudDetector, add_features,
                         rule_reasons, train_and_evaluate)

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE, "data", "sample_new_transactions.csv")
RISK_COLORS = {"High": "#d64545", "Medium": "#e3a008", "Low": "#2e9e6b"}

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Preparing data and training the AI models (first run only)...")
def load_everything():
    if not os.path.exists(DATA_PATH):
        from data_generator import generate
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)
        new = generate(n_customers=150, n_transactions=1000, seed=7, start="2026-10-01", days=14)
        new.drop(columns=["is_fraud"]).to_csv(SAMPLE_PATH, index=False)
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
        train_and_evaluate()
    det = FraudDetector.load()
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    data = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return det, metrics, data


det, metrics, data = load_everything()


def inr(x: float) -> str:
    """Indian-style compact amounts: ₹85.1 L, ₹1.2 Cr."""
    if x >= 1e7:
        return f"₹{x / 1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.1f} L"
    return f"₹{x:,.0f}"


def risk_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": " / 100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f2937"},
            "steps": [
                {"range": [0, det.threshold * 50], "color": "#c9eedb"},
                {"range": [det.threshold * 50, det.threshold * 100], "color": "#fbe7b5"},
                {"range": [det.threshold * 100, 100], "color": "#f6c4c4"},
            ],
            "threshold": {"line": {"color": "#d64545", "width": 4}, "value": det.threshold * 100},
        },
        title={"text": "Risk score"},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ FraudShield AI")
st.sidebar.caption("AI-Powered Fraud Detection System")
PAGES = {
    "dashboard": "📊 Dashboard",
    "check": "🔍 Check a Transaction",
    "batch": "📁 Batch Scan (CSV / Excel)",
    "performance": "📈 Model Performance",
    "about": "🤖 How the AI Works",
}
# direct links such as  http://localhost:8501/?page=check&example=suspicious
_start = list(PAGES).index(st.query_params.get("page", "dashboard")) \
    if st.query_params.get("page") in PAGES else 0
page = st.sidebar.radio("Navigate", list(PAGES.values()), index=_start)
st.sidebar.divider()
st.sidebar.markdown(
    f"**Models:** Random Forest + Isolation Forest  \n"
    f"**Trained on:** {metrics['n_train']:,} transactions  \n"
    f"**Alert threshold:** {det.threshold * 100:.1f} / 100"
)


# =========================================================================== DASHBOARD
if page == "📊 Dashboard":
    st.title("📊 Transaction Overview")
    st.caption("Historical card transactions (Jul-Sep 2026) used to train the AI. "
               "Fraud labels come from confirmed customer complaints / chargebacks.")

    fraud = data[data["is_fraud"] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(data):,}")
    c2.metric("Confirmed fraud", f"{len(fraud):,}", f"{len(fraud) / len(data):.1%} of all",
              delta_color="off")
    c3.metric("Money lost to fraud", inr(fraud["amount"].sum()))
    c4.metric("Customers", f"{data['customer_id'].nunique():,}")

    st.subheader("Unusual patterns the AI learned from")
    left, right = st.columns(2)

    by_hour = data.groupby("hour")["is_fraud"].mean().mul(100).reset_index()
    fig = px.bar(by_hour, x="hour", y="is_fraud", labels={"is_fraud": "Fraud rate (%)", "hour": "Hour of day"},
                 title="Fraud rate by hour of day",color_discrete_sequence=["#d64545"])
    left.plotly_chart(fig, width="stretch")

    by_cat = (data.groupby("merchant_category")["is_fraud"].mean().mul(100)
              .sort_values().reset_index())
    fig = px.bar(by_cat, x="is_fraud", y="merchant_category", orientation="h",
                 labels={"is_fraud": "Fraud rate (%)", "merchant_category": ""},
                 title="Fraud rate by merchant category", color_discrete_sequence=["#4361ee"])
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    d = add_features(data).assign(Type=lambda x: np.where(x["is_fraud"] == 1, "Fraud", "Genuine"))
    fig = px.box(d, x="Type", y="amount_to_avg_ratio", color="Type", log_y=True, points=False,
                 color_discrete_map={"Fraud": "#d64545", "Genuine": "#2e9e6b"},
                 labels={"amount_to_avg_ratio": "Amount ÷ customer's usual spend (log scale)"},
                 title="Fraud amounts are many times the customer's normal spend")
    left.plotly_chart(fig, width="stretch")

    signals = pd.DataFrame({
        "Signal": ["New device", "International", "Night (12-6am)", "4+ txns in 24h"],
        "Genuine": [d.loc[d.is_fraud == 0, c].mean() * 100 for c in ["is_new_device", "is_foreign", "is_night"]]
                   + [(d.loc[d.is_fraud == 0, "txn_count_24h"] >= 4).mean() * 100],
        "Fraud": [d.loc[d.is_fraud == 1, c].mean() * 100 for c in ["is_new_device", "is_foreign", "is_night"]]
                 + [(d.loc[d.is_fraud == 1, "txn_count_24h"] >= 4).mean() * 100],
    }).melt(id_vars="Signal", var_name="Type", value_name="% of transactions")
    fig = px.bar(signals, x="Signal", y="% of transactions", color="Type", barmode="group",
                 color_discrete_map={"Fraud": "#d64545", "Genuine": "#2e9e6b"},
                 title="How often each red flag appears")
    right.plotly_chart(fig, width="stretch")

    st.subheader("Most recent confirmed fraud cases")
    st.dataframe(fraud.sort_values("timestamp", ascending=False).head(15), width="stretch",
                 hide_index=True)


# =========================================================================== SINGLE CHECK
elif page == "🔍 Check a Transaction":
    st.title("🔍 Check a Transaction in Real Time")
    st.caption("Enter the details of a card transaction. The AI scores it instantly and explains why.")

    PRESETS = {
        "normal": dict(amount=1200.0, avg=1500.0, cat="Groceries", ch="POS", hour=18, dist=4.0,
                       foreign=False, device=False, count=2),
        "suspicious": dict(amount=42000.0, avg=1500.0, cat="Electronics", ch="Online", hour=2,
                           dist=4200.0, foreign=True, device=True, count=5),
        "borderline": dict(amount=6500.0, avg=1800.0, cat="Travel", ch="Online", hour=23,
                           dist=900.0, foreign=False, device=False, count=1),
    }

    def apply_preset(name):
        for k, v in PRESETS[name].items():
            st.session_state[f"f_{k}"] = v

    if "f_amount" not in st.session_state:
        example = st.query_params.get("example", "normal")
        apply_preset(example if example in PRESETS else "normal")

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    b1.button("✅ Normal example", on_click=apply_preset, args=("normal",))
    b2.button("⚠️ Borderline example", on_click=apply_preset, args=("borderline",))
    b3.button("🚨 Suspicious example", on_click=apply_preset, args=("suspicious",))

    form, result = st.columns([1, 1], gap="large")
    with form:
        c1, c2 = st.columns(2)
        amount = c1.number_input("Amount (₹)", min_value=1.0, step=100.0, key="f_amount")
        avg = c2.number_input("Customer's usual spend (₹)", min_value=1.0, step=100.0, key="f_avg",
                              help="Average transaction amount for this customer")
        cat = c1.selectbox("Merchant category", CATEGORIES, key="f_cat")
        ch = c2.selectbox("Channel", CHANNELS, key="f_ch")
        hour = c1.slider("Hour of day", 0, 23, key="f_hour")
        count = c2.slider("Transactions in last 24h", 1, 15, key="f_count")
        dist = st.number_input("Distance from home (km)", min_value=0.0, step=10.0, key="f_dist")
        c1, c2 = st.columns(2)
        foreign = c1.checkbox("International transaction", key="f_foreign")
        device = c2.checkbox("New / unrecognised device", key="f_device")

    row = add_features(pd.DataFrame([{
        "amount": amount, "customer_avg_amount": avg, "merchant_category": cat, "channel": ch,
        "hour": hour, "distance_from_home_km": dist, "is_foreign": int(foreign),
        "is_new_device": int(device), "txn_count_24h": count,
    }]))
    p_rf = float(det.rf_proba(row)[0])
    p_iso = float(det.anomaly_score(row)[0])
    score = float(det.risk_score(row)[0])
    level = det.risk_level(score)

    with result:
        st.plotly_chart(risk_gauge(score), width="stretch")
        verdict = {
            "High": ("🚨 HIGH RISK - block the transaction and alert the customer", st.error),
            "Medium": ("⚠️ MEDIUM RISK - ask the customer to verify (OTP / call)", st.warning),
            "Low": ("✅ LOW RISK - approve the transaction", st.success),
        }[level]
        verdict[1](verdict[0])
        m1, m2 = st.columns(2)
        m1.metric("Fraud probability (Random Forest)", f"{p_rf:.0%}")
        m2.metric("Anomaly score (Isolation Forest)", f"{p_iso:.0%}")

    st.subheader("Why did the AI give this score?")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Red flags found**")
        reasons = rule_reasons(row.iloc[0])
        if reasons:
            for r in reasons:
                st.markdown(f"- 🔴 {r}")
        else:
            st.markdown("- 🟢 No unusual patterns - looks like normal behaviour")
    with e2:
        contrib = pd.DataFrame(det.explain(row), columns=["Factor", "Impact"])
        contrib["Impact"] = (contrib["Impact"] * 100).round(1)
        contrib = contrib[contrib["Impact"].abs() >= 0.5]
        if len(contrib):
            contrib["Effect"] = np.where(contrib["Impact"] > 0, "Raises risk", "Lowers risk")
            fig = px.bar(contrib.iloc[::-1], x="Impact", y="Factor", orientation="h", color="Effect",
                         color_discrete_map={"Raises risk": "#d64545", "Lowers risk": "#2e9e6b"},
                         title="AI explanation (Shapley values)",
                         labels={"Impact": "Contribution to fraud probability (% points)", "Factor": ""})
            fig.update_layout(height=340, margin=dict(t=40, b=10), legend_title_text="",
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Every factor looks like this customer's normal behaviour.")


# =========================================================================== BATCH
elif page == "📁 Batch Scan (CSV / Excel)":
    st.title("📁 Scan a Batch of Transactions")
    st.caption("Upload a CSV or Excel file of new transactions (e.g. yesterday's activity). The AI scores every "
               "row and lists the ones investigators should look at first.")

    with st.expander("Required columns"):
        st.code("amount, customer_avg_amount, merchant_category, channel, hour,\n"
                "distance_from_home_km, is_foreign, is_new_device, txn_count_24h")
        st.caption(f"merchant_category: {', '.join(CATEGORIES)}  \nchannel: {', '.join(CHANNELS)}")

    up = st.file_uploader("Upload transactions (CSV or Excel)", type=["csv", "xlsx"],
                          help="For Excel files the first sheet is read; row 1 must hold the column names.")
    use_sample = st.toggle("Use the built-in sample file (1,000 new, unseen transactions)", value=up is None)

    raw = None
    if up is not None and not use_sample:
        try:
            raw = pd.read_excel(up) if up.name.lower().endswith(".xlsx") else pd.read_csv(up)
        except Exception as e:  # unreadable / corrupt file
            st.error(f"Could not read the file: {e}")
            st.stop()
    elif use_sample:
        raw = pd.read_csv(SAMPLE_PATH)

    if raw is None:
        st.info("Upload a file or switch on the sample file.")
    else:
        try:
            scored = det.score_frame(raw)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        high = scored[scored["risk_level"] == "High"]
        med = scored[scored["risk_level"] == "Medium"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions scanned", f"{len(scored):,}")
        c2.metric("🚨 High risk (flagged)", f"{len(high):,}")
        c3.metric("⚠️ Medium risk (verify)", f"{len(med):,}")
        c4.metric("Amount at risk (high)", inr(high["amount"].sum()))

        left, right = st.columns([1, 2])
        counts = scored["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55, color=counts.index,
                     color_discrete_map=RISK_COLORS, title="Risk distribution")
        left.plotly_chart(fig, width="stretch")
        fig = px.scatter(scored, x="hour", y="amount_to_avg_ratio", color="risk_level", log_y=True,
                         color_discrete_map=RISK_COLORS, hover_data=["amount", "merchant_category"],
                         labels={"amount_to_avg_ratio": "Amount ÷ usual spend", "hour": "Hour of day"},
                         title="Each dot is a transaction")
        right.plotly_chart(fig, width="stretch")

        show = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
        cols = [c for c in ["transaction_id", "customer_id", "timestamp", "amount", "merchant_category",
                            "channel", "hour", "risk_score", "risk_level", "reasons"] if c in scored.columns]
        view = scored[scored["risk_level"].isin(show)].sort_values("risk_score", ascending=False)
        st.dataframe(
            view[cols], width="stretch", hide_index=True,
            column_config={"risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0.0, max_value=1.0, format="%.2f")},
        )
        st.download_button("⬇️ Download scored results (CSV)", scored.to_csv(index=False),
                           file_name="fraud_scan_results.csv", mime="text/csv")


# =========================================================================== PERFORMANCE
elif page == "📈 Model Performance":
    st.title("📈 How Accurate is the AI?")
    st.caption(f"Tested on {metrics['n_test']:,} transactions the models never saw during training.")

    h = metrics["hybrid"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud caught (recall)", f"{h['recall']:.1%}")
    c2.metric("Alerts that are real fraud (precision)", f"{h['precision']:.1%}")
    c3.metric("F1 score", f"{h['f1']:.3f}")
    c4.metric("Fraud money caught",
              f"{metrics['fraud_amount_caught'] / metrics['fraud_amount_total']:.1%}")

    left, right = st.columns(2)
    cm = h["confusion"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted genuine", "Predicted fraud"], y=["Actually genuine", "Actually fraud"],
                    title="Confusion matrix (hybrid model)")
    fig.update_layout(coloraxis_showscale=False)
    left.plotly_chart(fig, width="stretch")

    roc = metrics["roc_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Hybrid AI",
                             line=dict(color="#4361ee", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(title=f"ROC curve (AUC = {h['roc_auc']:.3f})",
                      xaxis_title="False positive rate", yaxis_title="True positive rate")
    right.plotly_chart(fig, width="stretch")

    st.subheader("Comparing the two AI models")
    comp = pd.DataFrame({
        name: {"Precision": metrics[k]["precision"], "Recall": metrics[k]["recall"],
               "F1 score": metrics[k]["f1"], "ROC AUC": metrics[k]["roc_auc"],
               "PR AUC": metrics[k]["pr_auc"]}
        for name, k in [("Random Forest (supervised)", "random_forest"),
                        ("Isolation Forest (anomaly)", "isolation_forest"),
                        ("Hybrid (final system)", "hybrid")]
    }).T
    st.dataframe(comp.style.format("{:.3f}").highlight_max(
        axis=0, props="background-color:#2e9e6b;color:white;font-weight:bold"), width="stretch")

    st.subheader("Which signals matter most to the AI?")
    fi = pd.Series(metrics["feature_importance"]).sort_values()
    fig = px.bar(x=fi.values, y=fi.index, orientation="h", labels={"x": "Importance", "y": ""},
                 color_discrete_sequence=["#4361ee"])
    st.plotly_chart(fig, width="stretch")


# =========================================================================== ABOUT
else:
    st.title("🤖 How the AI Works")
    st.markdown(f"""
### The problem
Card fraud costs banks and customers crores every year. Fraud is **rare** (about
{metrics['fraud_rate']:.0%} of transactions here) and keeps changing, so fixed rules such as
*"block everything above ₹50,000"* either miss fraud or block genuine customers.

### The AI solution - a hybrid of two models

| | **Random Forest** | **Isolation Forest** |
|---|---|---|
| Type of AI | Supervised machine learning | Unsupervised anomaly detection |
| Learns from | Past transactions labelled fraud / genuine | Genuine transactions only |
| Answers | *"How similar is this to fraud I have seen?"* | *"How unusual is this compared to normal behaviour?"* |
| Strength | Very accurate on known fraud patterns | Can catch **new** fraud patterns |

**Final risk score = 75% × Random Forest probability + 25% × anomaly score**

### Step by step
1. **Data** - {metrics['n_train'] + metrics['n_test']:,} card transactions with amount, time, location,
   device, merchant and channel.
2. **Feature engineering** - raw fields become behavioural signals: amount ÷ the customer's usual
   spend, night-time flag, distance from home, number of transactions in 24 hours.
3. **Training** - the Random Forest (300 decision trees) learns fraud patterns; class weighting
   handles the fact that fraud is rare. The Isolation Forest learns normal behaviour.
4. **Threshold tuning** - the alert threshold is chosen on validation data to balance catching
   fraud against annoying genuine customers.
5. **Scoring and explanation** - each transaction gets a 0-100 risk score, a Low / Medium / High
   level and plain-language reasons (explainable AI).

### Actions for each risk level
- ✅ **Low** - approve automatically
- ⚠️ **Medium** - step-up verification (OTP, call to the customer)
- 🚨 **High** - block the transaction, freeze the card, alert the fraud team

### Where this is used
Banks and card networks for real-time card authorisation · UPI / wallet apps · e-commerce
checkout · insurance claims · internal audit of expense reimbursements.
""")
