"""SHAP engine (§8). Computed offline once, cached as explain.json — never at request time.

All models are explained at the level of the 20 cleaned features. For one-hot models the
SHAP values of a feature's dummy columns are summed, which is exact by SHAP's additivity, so
every model's panel speaks the same vocabulary and the Compare tab can line them up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .contract import write_json

CAVEAT = ("SHAP shows what the model uses, not what causes subscription. A feature can have high "
          "SHAP importance and no causal relationship with the outcome.")
MACRO_SET = set(C.MACRO)
SAMPLE_CAPS = {"decision_tree": 5000, "catboost": 5000, "tabnet": 500, "tabpfn": 200}


def aggregate_onehot(shap_matrix: np.ndarray, columns, source_map: dict, features: list[str]) -> np.ndarray:
    """Sum one-hot columns' SHAP values back onto their source feature."""
    out = np.zeros((shap_matrix.shape[0], len(features)))
    pos = {f: i for i, f in enumerate(features)}
    for j, col in enumerate(columns):
        out[:, pos[source_map[col]]] += shap_matrix[:, j]
    return out


def batched_kernel_shap(predict, X: np.ndarray, background: np.ndarray, n_coalitions: int, seed: int = C.RANDOM_STATE,
                        chunk: int = 8000):
    """Kernel SHAP (Lundberg & Lee 2017) with every model evaluation done in a few large batches.

    Same estimator as shap.KernelExplainer — coalitions sampled from the Shapley kernel, paired with their
    complements, values estimated by constrained least squares so that sum(phi) = f(x) - E[f] exactly —
    but it calls `predict` on ~chunk rows at a time instead of once per explained row. That matters for
    models with a large fixed cost per call (TabPFN re-reads its whole context on every call).
    Returns (phi [n × M], base, f(x) [n])."""
    rng = np.random.default_rng(seed)
    n, M = X.shape
    k = np.arange(1, M)
    p_size = (M - 1) / (k * (M - k))
    p_size /= p_size.sum()
    half = n_coalitions // 2
    Z = np.zeros((n, 2 * half, M), dtype=bool)
    for i in range(n):
        for s in range(half):
            size = rng.choice(k, p=p_size)
            z = np.zeros(M, bool)
            z[rng.choice(M, size, replace=False)] = True
            Z[i, 2 * s], Z[i, 2 * s + 1] = z, ~z
    B = len(background)
    # rows: for each (i, s, b): x_i where z, background_b elsewhere
    masked = np.where(Z[:, :, None, :], X[:, None, None, :], background[None, None, :, :]).reshape(-1, M)
    allrows = np.vstack([masked, X, background])
    preds = np.concatenate([predict(allrows[j:j + chunk]) for j in range(0, len(allrows), chunk)])
    v = preds[:len(masked)].reshape(n, 2 * half, B).mean(axis=2)
    fx = preds[len(masked):len(masked) + n]
    base = float(preds[len(masked) + n:].mean())
    phi = np.zeros((n, M))
    for i in range(n):
        z = Z[i].astype(float)
        target = fx[i] - base
        y = v[i] - base - z[:, -1] * target          # eliminate the last feature via the efficiency constraint
        A = z[:, :-1] - z[:, [-1]]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        phi[i, :-1] = sol
        phi[i, -1] = target - sol.sum()
    return phi, base, fx


def _fmt(v):
    if isinstance(v, (float, np.floating, int, np.integer)):
        v = float(v)
        if v.is_integer():
            return f"{int(v):,}" if abs(v) >= 10000 else str(int(v))
        return f"{v:,.1f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def _direction(feature, x: pd.Series, s: np.ndarray, categorical: set):
    if feature in categorical or x.nunique() < 3:
        means = pd.Series(s).groupby(x.astype(str).to_numpy()).mean().sort_values()
        return "mixed", f"{feature}={means.index[-1]} pushes up most; {feature}={means.index[0]} pushes down most"
    rho = spearmanr(x.to_numpy(dtype=float), s).statistic
    if not np.isfinite(rho) or abs(rho) < 0.1:
        return "mixed", f"no consistent direction for {feature}"
    return ("positive", f"higher {feature} pushes the prediction up") if rho > 0 else \
           ("negative", f"higher {feature} pushes the prediction down")


def build_payload(model_id: str, display_name: str, explainer: str, shap_values: np.ndarray,
                  X_raw: pd.DataFrame, proba: np.ndarray, y_true: np.ndarray, base_value: float,
                  units: str, client_ids, categorical=C.CATEGORICAL, extra_note: str = "") -> dict:
    feats = list(X_raw.columns)
    cats = set(categorical)
    n = len(X_raw)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    share = mean_abs / mean_abs.sum()

    directions, dir_text = [], []
    for j in order:
        d, t = _direction(feats[j], X_raw.iloc[:, j], shap_values[:, j], cats)
        directions.append(d)
        dir_text.append(t)

    top = [feats[j] for j in order]
    ranks = {f: i + 1 for i, f in enumerate(top)}
    macro_share = float(sum(share[j] for j in range(len(feats)) if feats[j] in MACRO_SET))

    global_text = (f"{display_name} relies most on {top[0]} ({share[order[0]]:.0%} of total attribution), "
                   f"then {top[1]} and {top[2]}. {dir_text[0][0].upper() + dir_text[0][1:]}.")

    # Beeswarm: top 12 features, ≤ 600 points
    rng = np.random.default_rng(C.RANDOM_STATE)
    pts = rng.choice(n, size=min(600, n), replace=False)
    bees = {"features": top[:12], "values": [], "feature_values": [], "labels": []}
    for j in order[:12]:
        col = X_raw.iloc[pts, j]
        bees["values"].append(np.round(shap_values[pts, j], 5).tolist())
        if feats[j] in cats:
            bees["feature_values"].append([None] * len(pts))
        else:
            r = col.rank(pct=True).to_numpy()
            bees["feature_values"].append(np.round(r, 3).tolist())
        bees["labels"].append([_fmt(v) for v in col])

    # Local examples: one of each kind a reader would ask about
    thr = np.quantile(proba, 1 - y_true.mean())
    picks = []

    def pick(mask, key, label):
        idx = np.where(mask)[0]
        if len(idx):
            i = idx[np.argmax(key[idx])]
            if i not in [p for p, _ in picks]:
                picks.append((i, label))

    pick(y_true == 1, proba, "Confident and right: a subscriber the model ranked highly")
    pick(y_true == 0, -proba, "Confident and right: a clear non-subscriber")
    pick((y_true == 0) & (proba >= thr), proba, "False alarm: ranked highly, did not subscribe")
    pick((y_true == 1) & (proba < thr), -proba, "Missed: subscribed, but the model ranked them low")
    pick(np.ones(n, bool), -np.abs(proba - thr), "Borderline: right at the decision cut-off")
    if "poutcome" in feats:
        pick((X_raw["poutcome"].astype(str) == "success").to_numpy() & (y_true == 1), proba,
             "Previous campaign succeeded")

    local = []
    for i, label in picks:
        contrib = shap_values[i]
        top_j = np.argsort(-np.abs(contrib))[:8]
        items = [{"feature": f"{feats[j]}={_fmt(X_raw.iloc[i, j])}", "value": float(contrib[j])} for j in top_j]
        rest = float(contrib.sum() - contrib[top_j].sum())
        items.append({"feature": f"{len(feats) - len(top_j)} other features", "value": rest})
        up = [it for it in items[:-1] if it["value"] > 0]
        down = [it for it in items[:-1] if it["value"] < 0]
        verdict = "likely" if proba[i] >= thr else "unlikely"
        txt = f"Scored {proba[i]:.0%} — ranked {verdict} to subscribe (actually {'did' if y_true[i] else 'did not'}). "
        if up:
            txt += f"Main push up: {up[0]['feature']}. "
        if down:
            txt += f"Main push down: {down[0]['feature']}."
        local.append({"client_id": int(client_ids[i]), "label": label, "prediction": float(proba[i]),
                      "y_true": int(y_true[i]), "base_value": float(base_value), "contributions": items,
                      "plain_text": txt.strip()})

    dow = ranks.get("day_of_week")
    body = (f"The five macro-economic indicators together account for {macro_share:.0%} of the model's "
            f"attribution — it is largely reading the state of the economy at the time of the call. "
            f"{top[0]}, {top[1]} and {top[2]} are the top three. ")
    if dow:
        body += (f"day_of_week ranks {dow} of {len(feats)} ({share[feats.index('day_of_week')]:.1%} of attribution), "
                 "consistent with its nearly flat subscription rate in the EDA. ")
    body += extra_note

    return {
        "kind": "shap",
        "model_id": model_id,
        "explainer": explainer,
        "sample_size": int(n),
        "units": units,
        "base_value": float(base_value),
        "global": {
            "features": top,
            "mean_abs_shap": [float(mean_abs[j]) for j in order],
            "share": [float(share[j]) for j in order],
            "direction": directions,
            "direction_text": dir_text,
            "macro_share": macro_share,
            "plain_text": global_text,
        },
        "beeswarm": bees,
        "local_examples": local,
        "narrative": {
            "headline": f"{display_name} is driven mainly by {top[0]}, {top[1]} and {top[2]}.",
            "body": body.strip(),
            "caveat": CAVEAT,
        },
    }


def write_explain(model_id: str, payload: dict):
    write_json(C.RESULTS_MODELS / model_id / "explain.json", payload, ndigits=5)
