"""Shared harness for the four supervised models: run with and without duration, score on the
same test set, write metrics.json / curves.json / predictions.json in the §6 shape."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .. import config as C
from ..contract import build_metrics, model_dir, write_json
from ..data_loader import Dataset, load_processed
from ..evaluation import best_f1_threshold, compute_metrics, curves


def run_both(model_id: str, encoding: str, fit_predict: Callable[[Dataset], dict],
             protocol_extra: dict | None = None) -> dict:
    """fit_predict(ds) must return:
        proba_test, thr_y, thr_proba (train-side predictions for choosing the threshold),
        train_seconds, predict_seconds, and optionally cv_roc_auc_mean / cv_roc_auc_std / model / extra.
    Returns {"without_duration": (ds, out), "with_duration": (ds, out)} for further use."""
    runs, outs, preds = {}, {}, {}
    for key, inc in [("without_duration", False), ("with_duration", True)]:
        ds = load_processed(include_duration=inc, encoding=encoding)
        out = fit_predict(ds)
        thr = best_f1_threshold(out["thr_y"], out["thr_proba"])
        m = compute_metrics(ds.y_test, out["proba_test"], threshold=thr)
        m.update(train_seconds=round(out["train_seconds"], 3), predict_seconds=round(out["predict_seconds"], 3),
                 cv_roc_auc_mean=out.get("cv_roc_auc_mean"), cv_roc_auc_std=out.get("cv_roc_auc_std"))
        runs[key] = m
        outs[key] = (ds, out)
        preds[key] = {"proba": np.round(out["proba_test"], 5).tolist(),
                      "pred": (out["proba_test"] >= thr).astype(int).tolist()}
        print(f"  [{model_id}] {key}: AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} thr={thr:.3f}")

    ds0 = outs["without_duration"][0]
    d = model_dir(model_id)
    write_json(d / "metrics.json", build_metrics(model_id, runs, ds0.y_test, protocol_extra))
    write_json(d / "curves.json", {k: curves(outs[k][0].y_test, outs[k][1]["proba_test"]) for k in outs})
    write_json(d / "predictions.json", {"test_index": ds0.test_index.tolist(), "y_true": ds0.y_test.tolist(), **preds})
    if runs["without_duration"]["roc_auc"] > C.AUC_RED_FLAG:
        raise RuntimeError("AUC without duration above red flag — audit the feature set")
    return outs


def params_block(items: list[tuple[str, object, str]]) -> dict:
    return {"params": [{"name": n, "value": v, "why": w} for n, v, w in items]}
