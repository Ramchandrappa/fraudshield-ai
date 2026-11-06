"""
Synthetic credit/debit card transaction generator.

Real bank transaction data is confidential, so this module creates a realistic
dataset of customer transactions in which a small share (~3%) are fraudulent.
Fraud is injected using patterns seen in real-world card fraud:
  * unusually large amounts compared with the customer's normal spending
  * transactions late at night
  * transactions far from the customer's home city / from abroad
  * a new or unrecognised device
  * rapid bursts of transactions (card-testing / spending sprees)
  * high-risk merchant categories (gift cards, crypto, electronics)

Not every fraudulent transaction shows every signal, and some genuine
transactions look unusual (travel, big purchases), so the AI must learn
combinations of signals rather than one simple rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "Groceries", "Restaurants", "Fuel", "Utilities", "Clothing",
    "Travel", "Electronics", "Online Shopping", "Gift Cards", "Crypto Exchange",
]
HIGH_RISK_CATEGORIES = {"Electronics", "Gift Cards", "Crypto Exchange", "Online Shopping"}
NORMAL_CAT_P = [0.25, 0.17, 0.12, 0.08, 0.10, 0.05, 0.06, 0.13, 0.02, 0.02]
FRAUD_CAT_P = [0.04, 0.04, 0.03, 0.01, 0.06, 0.08, 0.22, 0.20, 0.16, 0.16]
CHANNELS = ["POS", "Online", "ATM"]
CITIES = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata"]

# Hour-of-day weights for genuine activity (people mostly spend 8am-10pm)
_NORMAL_HOUR_W = np.array(
    [1, 0.5, 0.3, 0.2, 0.2, 0.5, 2, 4, 6, 7, 8, 9, 10, 9, 8, 8, 8, 9, 10, 10, 9, 7, 4, 2],
    dtype=float,
)
_NORMAL_HOUR_P = _NORMAL_HOUR_W / _NORMAL_HOUR_W.sum()


def _make_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "home_city": rng.choice(CITIES, n),
        # typical spend per transaction in INR (log-normal: most ~1,500, a few big spenders)
        "typical_spend": np.round(rng.lognormal(np.log(1500), 0.6, n), 2),
    })


def _genuine(cust: pd.DataFrame, n: int, start: pd.Timestamp, days: int,
             rng: np.random.Generator) -> pd.DataFrame:
    c = cust.sample(n, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    day = rng.integers(0, days, n)
    hour = rng.choice(24, n, p=_NORMAL_HOUR_P)
    minute = rng.integers(0, 60, n)
    ts = start + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(minute, "m")

    travelling = rng.random(n) < 0.04          # legitimate travel
    foreign = travelling & (rng.random(n) < 0.3)
    distance = np.where(travelling, rng.uniform(150, 2500, n), rng.exponential(6, n))
    distance = np.where(foreign, rng.uniform(2500, 9000, n), distance)

    amount = c["typical_spend"].values * rng.lognormal(0, 0.55, n)
    big_buy = rng.random(n) < 0.02              # occasional genuine big purchase
    amount = np.where(big_buy, amount * rng.uniform(3, 8, n), amount)

    return pd.DataFrame({
        "customer_id": c["customer_id"],
        "timestamp": ts,
        "amount": np.round(np.maximum(amount, 10), 2),
        "merchant_category": rng.choice(CATEGORIES, n, p=NORMAL_CAT_P),
        "channel": rng.choice(CHANNELS, n, p=[0.6, 0.3, 0.1]),
        "distance_from_home_km": np.round(distance, 1),
        "is_foreign": foreign.astype(int),
        "is_new_device": (rng.random(n) < 0.03).astype(int),
        "is_fraud": 0,
    })


def _fraud(cust: pd.DataFrame, n_target: int, start: pd.Timestamp, days: int,
           rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    while len(rows) < n_target:
        c = cust.iloc[int(rng.integers(len(cust)))]
        burst = int(rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.2, 0.2, 0.15, 0.1]))
        night = rng.random() < 0.6
        hour = int(rng.integers(0, 6)) if night else int(rng.integers(0, 24))
        t0 = start + pd.Timedelta(days=int(rng.integers(0, days)), hours=hour,
                                  minutes=int(rng.integers(0, 60)))
        far = rng.random() < 0.6
        foreign = far and rng.random() < 0.55
        new_device = rng.random() < 0.65
        channel = rng.choice(CHANNELS, p=[0.15, 0.65, 0.20])
        for k in range(burst):
            if rng.random() < 0.7:
                amt = c["typical_spend"] * rng.uniform(3, 15)        # big theft
            else:
                amt = c["typical_spend"] * rng.lognormal(0, 0.5)    # blends in
            dist = (rng.uniform(3000, 9000) if foreign else rng.uniform(200, 2500)) if far \
                else rng.exponential(8)
            rows.append({
                "customer_id": c["customer_id"],
                "timestamp": t0 + pd.Timedelta(minutes=int(k * rng.integers(1, 20))),
                "amount": round(max(float(amt), 10.0), 2),
                "merchant_category": rng.choice(CATEGORIES, p=FRAUD_CAT_P),
                "channel": channel,
                "distance_from_home_km": round(float(dist), 1),
                "is_foreign": int(foreign),
                "is_new_device": int(new_device),
                "is_fraud": 1,
            })
    return pd.DataFrame(rows[:n_target])


def _add_history_features(df: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    """Behavioural features computed from each customer's own history."""
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    typical = df["customer_id"].map(cust.set_index("customer_id")["typical_spend"])

    # customer's average amount over their previous (genuine-looking) transactions;
    # the first transaction falls back to the profile's typical spend
    prev_mean = df.groupby("customer_id")["amount"].transform(
        lambda s: s.shift().expanding().median())
    df["customer_avg_amount"] = np.round(prev_mean.fillna(typical), 2)

    # number of the customer's transactions in the last 24 hours (including this one)
    counts = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("customer_id").indices.items():
        t = df.loc[idx, "timestamp"].values.astype("datetime64[m]").astype(np.int64)
        left = np.searchsorted(t, t - 24 * 60, side="right")
        counts[idx] = np.arange(len(t)) - left + 1
    df["txn_count_24h"] = counts
    df["hour"] = df["timestamp"].dt.hour
    return df


def generate(n_customers: int = 600, n_transactions: int = 25000, fraud_rate: float = 0.03,
             days: int = 90, seed: int = 42, start: str = "2026-07-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    cust = _make_customers(n_customers, rng)
    n_fraud = int(n_transactions * fraud_rate)
    df = pd.concat([
        _genuine(cust, n_transactions - n_fraud, start_ts, days, rng),
        _fraud(cust, n_fraud, start_ts, days, rng),
    ], ignore_index=True)
    df = _add_history_features(df, cust)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i:06d}" for i in range(1, len(df) + 1)])
    cols = ["transaction_id", "customer_id", "timestamp", "amount", "customer_avg_amount",
            "merchant_category", "channel", "hour", "distance_from_home_km", "is_foreign",
            "is_new_device", "txn_count_24h", "is_fraud"]
    return df[cols]


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    data = generate()
    data.to_csv("data/transactions.csv", index=False)
    # a fresh, unseen batch (different seed) that users can upload in the app
    new = generate(n_customers=150, n_transactions=1000, seed=7, start="2026-10-01", days=14)
    new.drop(columns=["is_fraud"]).to_csv("data/sample_new_transactions.csv", index=False)
    print(f"transactions.csv: {len(data):,} rows, fraud rate {data['is_fraud'].mean():.2%}")
    print(f"sample_new_transactions.csv: {len(new):,} rows (unlabelled)")
