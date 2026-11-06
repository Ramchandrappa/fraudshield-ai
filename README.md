# FraudShield AI: AI-Powered Fraud Detection System

A web app that uses AI (a Random Forest combined with an Isolation Forest) to score card
transactions, flag suspicious ones based on unusual patterns, and explain each decision.
The full write-up is in [PROJECT_REPORT.md](PROJECT_REPORT.md).

## How to run

```bash
pip install -r requirements.txt
```

```bash
python -m streamlit run app.py
```

Then open http://localhost:8501 in your browser.

The data and trained model are already in `data/` and `models/`. To rebuild them from scratch:

```bash
python data_generator.py
```

```bash
python fraud_model.py
```

(If those folders are deleted, the app regenerates and retrains automatically on first launch.)

## Demo script for the presentation

1. **Dashboard:** show the fraud patterns: night hours, gift card and crypto merchants, large amounts.
2. **Check a Transaction:** click *Normal* (low risk), then *Borderline* (medium, verify with OTP), then *Suspicious* (high, block). Explain the red flags and the Shapley chart.
3. **Batch Scan:** scan the 1,000-transaction sample, show the ranked high-risk list, download the results.
4. **Model Performance:** 87.7% of fraud caught, 90.6% of alerts correct, ROC AUC 0.998, and the hybrid beats each model alone.
5. **How the AI Works:** summarise the approach and its real-world uses.
