# FraudShield AI: AI-Powered Fraud Detection System

A web app that uses AI (a Random Forest combined with an Isolation Forest) to score card
transactions, flag suspicious ones based on unusual patterns, and explain each decision.

## Reports

| Report | Format |
|---|---|
| Step-by-step project report (19 pages, with app screenshots) | [PDF](FraudShield_AI_Step_by_Step_Report.pdf) · [Word](FraudShield_AI_Step_by_Step_Report.docx) |
| Short project report (7 pages) | [PDF](FraudShield_AI_Project_Report.pdf) · [Word](FraudShield_AI_Project_Report.docx) · [Markdown](PROJECT_REPORT.md) |

## Results (6,250 unseen test transactions)

| Fraud caught (recall) | Alerts that were real fraud (precision) | Fraud money caught | Genuine customers wrongly flagged |
|---|---|---|---|
| 87.7% | 90.6% | 96.5% | 0.3% |

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

## Test data

- [`data/test_10_customers.xlsx`](data/test_10_customers.xlsx): 10 customer transactions with expected results. Upload it on the **Batch Scan** page and you should get 5 Low, 2 Medium and 3 High.
- `data/sample_new_transactions.csv`: 1,000 new transactions, built into the Batch Scan page.

## Direct links to app pages

Add `?page=` to the app address to open a page directly: `dashboard`, `check`, `batch`, `performance` or `about`.
On the Check page, add `&example=normal`, `borderline` or `suspicious`.

For example: http://localhost:8501/?page=check&example=suspicious

## Demo script for the presentation

1. **Dashboard:** show the fraud patterns: night hours, gift card and crypto merchants, large amounts.
2. **Check a Transaction:** click *Normal* (low risk), then *Borderline* (medium, verify with OTP), then *Suspicious* (high, block). Explain the red flags and the Shapley chart.
3. **Batch Scan:** scan the 1,000-transaction sample, show the ranked high-risk list, download the results.
4. **Model Performance:** 87.7% of fraud caught, 90.6% of alerts correct, ROC AUC 0.998, and the hybrid beats each model alone.
5. **How the AI Works:** summarise the approach and its real-world uses.
