# FraudShield AI: AI-Powered Fraud Detection System

**Course:** AI for Finance, CIA-2: AI-Powered Project Development
**Project option:** 3. AI-Powered Fraud Detection System

---

## 1. Problem statement

Card and digital-payment fraud is a growing problem for banks, payment apps and their customers.
Fraud is hard to catch for three reasons:

- **It is rare.** In this dataset only about 3% of transactions are fraud, so a model that simply says "genuine" every time is 97% "accurate" and completely useless.
- **It keeps changing.** Fraudsters adapt, so fixed rules such as "block everything above ₹50,000" quickly go out of date.
- **Mistakes are costly both ways.** A missed fraud loses money. A false alarm blocks a genuine customer's card and annoys them.

**Goal:** Build a system that analyses each financial transaction, gives it a fraud risk score,
flags suspicious transactions based on unusual patterns, and explains *why* each one was flagged.

## 2. Solution overview

FraudShield AI is a web application built with Python. A **hybrid of two AI models** scores every
transaction from 0 to 100:

| | Random Forest | Isolation Forest |
|---|---|---|
| Type of AI | Supervised machine learning | Unsupervised anomaly detection |
| Learns from | Past transactions labelled fraud / genuine | Genuine transactions only |
| Question it answers | "How much does this look like fraud I've seen before?" | "How unusual is this compared to normal behaviour?" |
| Strength | High accuracy on known fraud patterns | Can catch new fraud patterns it was never shown |

**Final risk score = 0.75 × Random Forest probability + 0.25 × Isolation Forest anomaly score**

Each transaction then gets a risk level and an action:

| Risk level | Action |
|---|---|
| ✅ Low | Approve automatically |
| ⚠️ Medium | Step-up verification (OTP or a call to the customer) |
| 🚨 High | Block the transaction and alert the fraud team |

## 3. Data

Real bank transaction data is confidential, so the project uses a **realistic synthetic dataset**
(`data_generator.py`): 25,000 card transactions from 600 customers over 90 days (Jul–Sep 2026),
with 750 fraud cases (3%). Fraud was added using patterns documented in real-world card fraud:

- Amounts many times larger than the customer's usual spend
- Transactions late at night (12am–6am)
- Locations far from home or abroad
- A new or unrecognised device
- Bursts of several transactions in a short time (card testing or spending sprees)
- High-risk merchants (gift cards, crypto exchanges, electronics)

To keep the problem realistic, not every fraud shows every signal. Some fraud amounts look
normal, and some genuine transactions look unusual (travel, occasional big purchases).

| Field | Meaning |
|---|---|
| amount | Transaction amount (₹) |
| customer_avg_amount | Customer's usual transaction amount, from their history |
| merchant_category | Groceries, Travel, Electronics, Gift Cards, … |
| channel | POS (card swipe), Online, ATM |
| hour | Hour of day (0–23) |
| distance_from_home_km | Distance from the customer's home city |
| is_foreign | 1 = international transaction |
| is_new_device | 1 = device not seen before for this customer |
| txn_count_24h | Customer's transactions in the last 24 hours |
| is_fraud | Label: 1 = confirmed fraud |

## 4. How AI is used

1. **Feature engineering:** raw fields become behavioural signals. The most important is
   *amount ÷ customer's usual spend*, which compares each transaction with *that customer's* own
   behaviour rather than with a fixed limit. Other signals are a night-time flag, distance and
   transaction velocity.
2. **Random Forest classifier:** 300 decision trees vote on whether a transaction is fraud.
   `class_weight="balanced_subsample"` makes the model pay attention to the rare fraud class.
3. **Isolation Forest:** trained only on genuine transactions. It isolates points that are
   "easy to separate" from the normal crowd and gives them a high anomaly score.
4. **Threshold tuning:** the alert threshold (52.4 / 100) is chosen on a separate
   *validation* set to maximise the F1 score, which balances fraud caught against false alarms.
   The test set is never used for tuning.
5. **Explainable AI:**
   - *Red-flag reasons* in plain language, e.g. "Amount is 28× the customer's usual spend",
     "4,200 km from home".
   - *Shapley values* (from cooperative game theory, the idea behind SHAP). Each factor's
     contribution is measured by comparing the transaction with a typical genuine transaction
     across all 256 what-if combinations of the 8 factors. This shows exactly how much each
     factor raised or lowered the fraud probability.
6. **AI tools in development:** the project was designed and coded with the help of an AI
   coding assistant (Claude Code), which was used to plan the architecture, write and debug
   the code, and test the web app in a browser.

## 5. Results

The data was split 75% / 25%. All numbers below are on the **6,250 test transactions the models
never saw during training**.

| Model | Precision | Recall | F1 score | ROC AUC |
|---|---|---|---|---|
| Random Forest alone | 0.901 | 0.872 | 0.886 | 0.998 |
| Isolation Forest alone | 0.925 | 0.797 | 0.856 | 0.995 |
| **Hybrid (final system)** | **0.906** | **0.877** | **0.891** | **0.998** |

**Confusion matrix (hybrid):** 164 frauds caught, 23 missed, 17 false alarms, 6,046 genuine
transactions correctly approved.

- **87.7%** of fraud cases were caught (recall).
- **90.6%** of alerts were real fraud (precision), so investigators don't waste time.
- **96.5%** of the money involved in fraud was caught, because the large frauds are the easiest to detect.
- Only **0.3%** of genuine customers were wrongly flagged.
- On a separate batch of 1,000 brand-new transactions (the app's sample file), the system flagged 34, caught 28 of the 30 real frauds (93%) and 82% of its alerts were correct.

**Most important signals (Random Forest feature importance):** amount vs. usual spend > time of
day > new device > distance from home > night-time.

*Note: The data is synthetic, so real-world accuracy would be lower. Real data is noisier and
fraudsters actively try to look normal.*

## 6. Features of the application

| Page | What it does |
|---|---|
| 📊 Dashboard | KPIs (transactions, fraud cases, money lost) and charts showing fraud patterns by hour, merchant, amount and red flag |
| 🔍 Check a Transaction | Enter one transaction and get an instant risk score on a gauge, a decision (approve / verify / block), red-flag reasons and a Shapley explanation chart. Normal, borderline and suspicious examples are one click away. |
| 📁 Batch Scan (CSV) | Upload a CSV (or use the sample of 1,000 new transactions). Every row is scored and the list is sorted by risk with reasons. Results can be downloaded as CSV. |
| 📈 Model Performance | Confusion matrix, ROC curve, comparison of the three models and feature importance |
| 🤖 How the AI Works | Plain-language explanation of the approach |

## 7. Benefits and practical applications

- **Banks and card networks:** score card payments in real time, in milliseconds, before they are approved.
- **UPI, wallet and e-commerce platforms:** detect account takeover and payment fraud at checkout.
- **Insurance and corporate finance:** flag unusual claims or expense reimbursements for audit.
- **Benefits:** less money lost, fewer genuine customers blocked than with fixed rules, faster
  investigations (alerts are ranked and explained), and easier compliance (each decision is explainable).

## 8. Limitations and future work

- Retrain on real, anonymised bank data and refresh the model regularly as fraud patterns change.
- Add more signals: merchant reputation, IP address and device fingerprint, the customer's spending network.
- Try gradient boosting (XGBoost / LightGBM) and deep-learning sequence models.
- Use investigator feedback to keep improving the model (active learning).
- Deploy the scoring as an API connected to a live payment system.

## 9. Technology stack

Python 3.10 · scikit-learn (Random Forest, Isolation Forest) · pandas / NumPy · Streamlit (web UI) · Plotly (charts) · joblib (model saving)

## 10. Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application (user interface) |
| `fraud_model.py` | AI core: features, hybrid model, training, evaluation, explanations |
| `data_generator.py` | Builds the synthetic transaction dataset |
| `data/transactions.csv` | 25,000 labelled transactions (training and testing) |
| `data/sample_new_transactions.csv` | 1,000 unlabelled new transactions for the Batch Scan demo |
| `models/` | Saved trained model and evaluation metrics |
| `requirements.txt` | Python packages needed |
