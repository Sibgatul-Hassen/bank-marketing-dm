"""Reproduces the plan's §3.6 leakage table in our own pipeline: six classical model families,
each trained with and without duration on the identical split.   python -m src.models.reference_models

These are not model pages — they feed the Comparison page's leakage analysis."""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from .. import config as C
from ..contract import write_json
from ..data_loader import load_processed
from ..evaluation import compute_metrics, curves
from ..preprocessing import make_onehot_preprocessor

# Measured in the master plan §3.6 (AUC with, AUC without) — the regression test for this script.
PLAN = {"hist_gb": (0.9546, 0.8136), "random_forest": (0.9494, 0.7852), "log_reg": (0.9439, 0.8009),
        "decision_tree_d5": (0.9380, 0.7902), "knn_25": (0.8815, 0.7727), "gaussian_nb": (0.8393, 0.7755)}

MODELS = {
    "hist_gb": ("HistGradientBoosting", False, lambda: HistGradientBoostingClassifier(random_state=C.RANDOM_STATE)),
    "random_forest": ("Random Forest", False, lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=C.RANDOM_STATE)),
    "log_reg": ("Logistic Regression", True, lambda: LogisticRegression(max_iter=2000)),
    "decision_tree_d5": ("Decision Tree (d=5)", False, lambda: DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=C.RANDOM_STATE)),
    "knn_25": ("kNN (k=25)", True, lambda: KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
    "gaussian_nb": ("Gaussian NB", True, lambda: GaussianNB()),
}


def main():
    out = {}
    for inc in (False, True):
        ds = load_processed(include_duration=inc)
        key = "with_duration" if inc else "without_duration"
        for mid, (name, scale, make) in MODELS.items():
            pipe = make_pipeline(make_onehot_preprocessor(ds.categorical, ds.numeric, scale=scale), make())
            t0 = time.perf_counter()
            pipe.fit(ds.X_train, ds.y_train)
            t1 = time.perf_counter()
            p = pipe.predict_proba(ds.X_test)[:, 1]
            t2 = time.perf_counter()
            m = compute_metrics(ds.y_test, p, threshold=0.5)
            m.update(train_seconds=t1 - t0, predict_seconds=t2 - t1)
            entry = out.setdefault(mid, {"model_id": mid, "display_name": name, "runs": {}, "curves": {}, "pred": {}})
            entry["runs"][key] = m
            entry["curves"][key] = curves(ds.y_test, p)
            entry["pred"][key] = (p >= 0.5).astype(int).tolist()
            plan = PLAN[mid][0 if inc else 1]
            print(f"  {name:22s} {key:17s} AUC {m['roc_auc']:.4f} (plan {plan:.4f}, Δ {m['roc_auc'] - plan:+.4f})", flush=True)
    for mid, e in out.items():
        e["plan_auc"] = {"with_duration": PLAN[mid][0], "without_duration": PLAN[mid][1]}
        e["leak_delta"] = {k: e["runs"]["with_duration"][k] - e["runs"]["without_duration"][k] for k in ("roc_auc", "pr_auc", "f1")}
    write_json(C.RESULTS_COMPARISON / "reference_models.json", {
        "note": "Six classical families from the plan's §3.6, re-measured in this pipeline (same split, same cleaning). "
                "Default hyperparameters, threshold 0.5 — these exist to measure leak sensitivity, not to compete.",
        "models": list(out.values())})


if __name__ == "__main__":
    main()
