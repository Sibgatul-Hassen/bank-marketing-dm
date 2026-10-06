"""One shared metrics function. Every model page's numbers come from here."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)

from . import config as C

PROTOCOL = {
    "split": "stratified 80/20 holdout",
    "cv": "stratified 5-fold on train",
    "random_state": C.RANDOM_STATE,
    "threshold": "decision threshold chosen to maximise F1 on training-side predictions "
                 "(out-of-fold or validation), never on the test set",
    "note": "All scaling/encoding/resampling fitted inside the fold, never before the split.",
}


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximising F1. Call this on train/validation predictions only."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])]) if len(t) else 0.5


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def baselines(y_true) -> dict:
    y_true = np.asarray(y_true)
    return {"majority_class": {
        "accuracy": round(float((y_true == 0).mean()), 4),
        "roc_auc": 0.5,
        "pr_auc": round(float(y_true.mean()), 4),
        "f1": 0.0,
        "note": "Always predict 'no'. Every accuracy figure must be read against this row.",
    }}


def _downsample(xs, ys, n=200):
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) <= n:
        return xs, ys
    keep = np.unique(np.linspace(0, len(xs) - 1, n).round().astype(int))
    return xs[keep], ys[keep]


def curves(y_true, proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fpr, tpr = _downsample(fpr, tpr)
    rec, prec = _downsample(rec[::-1], prec[::-1])
    return {
        "roc": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
        "pr": {"recall": np.round(rec, 4).tolist(), "precision": np.round(prec, 4).tolist()},
        "lift": lift_curve(y_true, proba),
    }


def lift_curve(y_true, proba, points=20) -> dict:
    """Cumulative gains: share of all subscribers captured when calling the top q% by score."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    cum = np.cumsum(y_true[order]) / y_true.sum()
    qs = np.linspace(0, 1, points + 1)
    captured = [0.0] + [float(cum[max(int(round(q * len(cum))) - 1, 0)]) for q in qs[1:]]
    return {"fraction_called": np.round(qs, 3).tolist(), "fraction_captured": np.round(captured, 4).tolist()}


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar's test with continuity correction on paired test-set predictions."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}
