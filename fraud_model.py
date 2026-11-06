"""
AI core of the Fraud Detection System.

Two AI models work together (a "hybrid" approach used by real banks):

1. Random Forest classifier  (supervised machine learning)
   Learns from past transactions that were labelled fraud / genuine and
   outputs the probability that a new transaction is fraud.

2. Isolation Forest  (unsupervised anomaly detection)
   Learns what *normal* behaviour looks like and scores how unusual a new
   transaction is - so it can also catch new fraud patterns that never
   appeared in the training labels.

Final risk score = 0.75 x Random Forest probability + 0.25 x anomaly score.
Each flagged transaction is explained in plain language (explainable AI).
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_generator import CATEGORIES, CHANNELS, HIGH_RISK_CATEGORIES

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "fraud_model.joblib")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")
DATA_PATH = os.path.join(BASE, "data", "transactions.csv")

NUM_FEATURES = ["amount", "amount_to_avg_ratio", "hour", "is_night", "distance_from_home_km",
                "is_foreign", "is_new_device", "txn_count_24h"]
CAT_FEATURES = ["merchant_category", "channel"]
REQUIRED_COLUMNS = ["amount", "customer_avg_amount", "merchant_category", "channel", "hour",
                    "distance_from_home_km", "is_foreign", "is_new_device", "txn_count_24h"]

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "amount_to_avg_ratio": "Amount vs. customer's usual spend",
    "hour": "Time of day",
    "is_night": "Night-time (12am-6am)",
    "distance_from_home_km": "Distance from home",
    "is_foreign": "International transaction",
    "is_new_device": "New / unknown device",
    "txn_count_24h": "Transactions in last 24h",
    "merchant_category": "Merchant category",
    "channel": "Channel (POS / Online / ATM)",
}

RF_WEIGHT, ISO_WEIGHT = 0.75, 0.25


# --------------------------------------------------------------------------- features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: turn raw fields into behavioural signals."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    out = df.copy()
    for c in ["amount", "customer_avg_amount", "hour", "distance_from_home_km",
              "is_foreign", "is_new_device", "txn_count_24h"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS)
    out["amount_to_avg_ratio"] = out["amount"] / out["customer_avg_amount"].clip(lower=1)
    out["is_night"] = (out["hour"] < 6).astype(int)
    return out


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(categories=[CATEGORIES, CHANNELS], handle_unknown="ignore"),
         CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])


# --------------------------------------------------------------------------- model
class FraudDetector:
    def __init__(self):
        self.rf = Pipeline([
            ("prep", _preprocessor()),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2,
                                             class_weight="balanced_subsample",
                                             random_state=42, n_jobs=-1)),
        ])
        self.iso = Pipeline([
            ("prep", _preprocessor()),
            ("model", IsolationForest(n_estimators=300, contamination="auto", random_state=42,
                                      n_jobs=-1)),
        ])
        self.iso_min = 0.0
        self.iso_max = 1.0
        self.threshold = 0.5
        self.baseline: dict = {}

    # ---- training
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudDetector":
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                    random_state=1)
        self.rf.fit(X_tr, y_tr)
        # Isolation Forest only learns from genuine behaviour
        self.iso.fit(X_tr[y_tr == 0])
        raw = -self.iso.score_samples(X_tr)
        self.iso_min, self.iso_max = float(np.percentile(raw, 1)), float(np.percentile(raw, 99.5))

        # choose the alert threshold that gives the best F1 on held-out validation data
        # (the stand-alone models get their own thresholds, for a fair comparison)
        self.threshold = _best_threshold(y_val, self.risk_score(X_val))
        self.rf_threshold = _best_threshold(y_val, self.rf_proba(X_val))
        self.iso_threshold = _best_threshold(y_val, self.anomaly_score(X_val))

        # "typical genuine" values, used to explain individual predictions
        genuine = X_tr[y_tr == 0]
        self.baseline = {c: float(genuine[c].median()) for c in NUM_FEATURES}
        self.baseline["merchant_category"] = "Groceries"
        self.baseline["channel"] = "POS"

        # refit the classifier on all training data
        self.rf.fit(X, y)
        return self

    # ---- scoring
    def rf_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.rf.predict_proba(X)[:, 1]

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        raw = -self.iso.score_samples(X)
        return np.clip((raw - self.iso_min) / (self.iso_max - self.iso_min), 0, 1)

    def risk_score(self, X: pd.DataFrame) -> np.ndarray:
        return RF_WEIGHT * self.rf_proba(X) + ISO_WEIGHT * self.anomaly_score(X)

    def risk_level(self, score: float) -> str:
        if score >= self.threshold:
            return "High"
        if score >= self.threshold * 0.5:
            return "Medium"
        return "Low"

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of raw transactions and add explanation columns."""
        X = add_features(df)
        out = X.copy()
        out["fraud_probability"] = np.round(self.rf_proba(X), 4)
        out["anomaly_score"] = np.round(self.anomaly_score(X), 4)
        out["risk_score"] = np.round(RF_WEIGHT * out["fraud_probability"]
                                     + ISO_WEIGHT * out["anomaly_score"], 4)
        out["risk_level"] = out["risk_score"].apply(self.risk_level)
        out["flagged"] = (out["risk_score"] >= self.threshold).astype(int)
        out["reasons"] = [", ".join(r) if r else "-" for r in
                          (rule_reasons(row) for _, row in out.iterrows())]
        return out

    # ---- explainability
    EXPLAIN_GROUPS = ["amount", "hour", "distance_from_home_km", "is_foreign", "is_new_device",
                      "txn_count_24h", "merchant_category", "channel"]

    def explain(self, row: pd.DataFrame) -> list[tuple[str, float]]:
        """Exact Shapley values: each factor's fair share of the fraud probability.

        Compared with a 'typical genuine transaction' of the same customer, every
        combination of the 8 factors (2^8 = 256 what-if transactions) is scored, and
        game theory (Shapley values) splits the change in fraud probability between
        the factors. The values add up to: probability(this) - probability(typical).
        """
        from itertools import product
        from math import factorial

        groups = self.EXPLAIN_GROUPS
        n = len(groups)
        r = row.iloc[0]
        typical = {g: self.baseline[g] for g in groups}
        typical["amount"] = self.baseline["amount_to_avg_ratio"] * max(r["customer_avg_amount"], 1)

        masks = list(product([0, 1], repeat=n))
        rows = []
        for m in masks:
            d = {c: r[c] for c in REQUIRED_COLUMNS}
            for g, on in zip(groups, m):
                if not on:
                    d[g] = typical[g]
            rows.append(d)
        preds = self.rf_proba(add_features(pd.DataFrame(rows)))
        value = {m: p for m, p in zip(masks, preds)}

        phi = np.zeros(n)
        for m in masks:
            k = sum(m)
            for i in range(n):
                if m[i] == 0:
                    with_i = m[:i] + (1,) + m[i + 1:]
                    w = factorial(k) * factorial(n - k - 1) / factorial(n)
                    phi[i] += w * (value[with_i] - value[m])
        contrib = [(FEATURE_LABELS[g], float(v)) for g, v in zip(groups, phi)]
        return sorted(contrib, key=lambda x: x[1], reverse=True)

    def feature_importance(self) -> pd.Series:
        names = self.rf.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(self.rf.named_steps["model"].feature_importances_, index=names)
        grouped = {}
        for name, v in imp.items():
            kind, feat = name.split("__", 1)
            key = feat
            if kind == "cat":
                key = "merchant_category" if feat.startswith("merchant_category") else "channel"
            grouped[FEATURE_LABELS[key]] = grouped.get(FEATURE_LABELS[key], 0) + v
        return pd.Series(grouped).sort_values(ascending=False)

    # ---- persistence
    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "FraudDetector":
        return joblib.load(path)


def rule_reasons(r: pd.Series) -> list[str]:
    """Human-readable red flags for a single transaction."""
    reasons = []
    if r["amount_to_avg_ratio"] >= 3:
        reasons.append(f"Amount is {r['amount_to_avg_ratio']:.1f}x the customer's usual spend")
    if r["hour"] < 6:
        reasons.append(f"Unusual time ({int(r['hour']):02d}:00)")
    if r["distance_from_home_km"] >= 150:
        reasons.append(f"{r['distance_from_home_km']:,.0f} km from home")
    if r["is_foreign"] == 1:
        reasons.append("International transaction")
    if r["is_new_device"] == 1:
        reasons.append("New / unrecognised device")
    if r["txn_count_24h"] >= 4:
        reasons.append(f"{int(r['txn_count_24h'])} transactions in 24h (burst)")
    if r["merchant_category"] in HIGH_RISK_CATEGORIES:
        reasons.append(f"High-risk merchant: {r['merchant_category']}")
    return reasons


# --------------------------------------------------------------------------- training script
def _metrics(y, scores, threshold) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _best_threshold(y, scores) -> float:
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.argmax(f1[:-1])])


def train_and_evaluate(data_path: str = DATA_PATH) -> dict:
    df = add_features(pd.read_csv(data_path))
    y = df["is_fraud"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, stratify=y,
                                                        random_state=42)
    det = FraudDetector().fit(X_train, y_train)

    rf_s, iso_s = det.rf_proba(X_test), det.anomaly_score(X_test)
    hybrid = RF_WEIGHT * rf_s + ISO_WEIGHT * iso_s
    fpr, tpr, _ = roc_curve(y_test, hybrid)
    keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int)

    test_scored = X_test.assign(risk_score=hybrid, is_fraud=y_test.values)
    caught_amount = float(test_scored.loc[(hybrid >= det.threshold) & (y_test == 1), "amount"].sum())
    total_fraud_amount = float(test_scored.loc[y_test == 1, "amount"].sum())

    metrics = {
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "fraud_rate": float(y.mean()),
        "hybrid": _metrics(y_test, hybrid, det.threshold),
        # all thresholds were chosen on validation data, never on the test set
        "random_forest": _metrics(y_test, rf_s, det.rf_threshold),
        "isolation_forest": _metrics(y_test, iso_s, det.iso_threshold),
        "roc_curve": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        "feature_importance": det.feature_importance().round(4).to_dict(),
        "fraud_amount_caught": caught_amount,
        "fraud_amount_total": total_fraud_amount,
    }
    det.save()
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    # import through the module name so the saved model can be loaded by the app
    from fraud_model import train_and_evaluate as _train
    m = _train()
    for name in ["random_forest", "isolation_forest", "hybrid"]:
        s = m[name]
        print(f"{name:17s} precision={s['precision']:.3f} recall={s['recall']:.3f} "
              f"f1={s['f1']:.3f} roc_auc={s['roc_auc']:.3f} pr_auc={s['pr_auc']:.3f}")
    print("confusion (hybrid):", m["hybrid"]["confusion"])
    print("top features:", list(m["feature_importance"].items())[:5])
