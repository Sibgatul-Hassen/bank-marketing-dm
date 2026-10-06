"""Agent-02 — CatBoost (§7.2).   python -m src.models.catboost_model"""
from __future__ import annotations

import json
import time

import numpy as np
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .. import config as C
from ..contract import model_dir, validate_model_artifacts, write_figure, write_json
from ..evaluation import mcnemar
from ..explain import SAMPLE_CAPS, build_payload, write_explain
from ._supervised import params_block, run_both

MID = "catboost"
ACCENT, MUTED, AMBER, BLUE = "#4FB3A0", "#6B7E8F", "#D4A373", "#7B9FC7"
PARAMS = dict(iterations=1000, learning_rate=0.05, depth=6, eval_metric="AUC", loss_function="Logloss",
              auto_class_weights="Balanced", random_seed=C.RANDOM_STATE, thread_count=-1, verbose=False)
EARLY_STOP = 50
VAL_SIZE = 0.15


def make_model(**kw):
    return CatBoostClassifier(**{**PARAMS, **kw})


def fit_predict(ds):
    cats = ds.categorical
    X_tr, X_val, y_tr, y_val = train_test_split(ds.X_train, ds.y_train, test_size=VAL_SIZE,
                                                stratify=ds.y_train, random_state=C.RANDOM_STATE)
    model = make_model(custom_metric=["Logloss", "AUC"])
    t0 = time.perf_counter()
    model.fit(Pool(X_tr, y_tr, cat_features=cats), eval_set=Pool(X_val, y_val, cat_features=cats),
              early_stopping_rounds=EARLY_STOP, use_best_model=True)
    t1 = time.perf_counter()
    proba = model.predict_proba(Pool(ds.X_test, cat_features=cats))[:, 1]
    t2 = time.perf_counter()
    val_proba = model.predict_proba(Pool(X_val, cat_features=cats))[:, 1]

    # CV AUC on the training split at the early-stopped iteration count
    best_it = model.get_best_iteration() + 1
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    aucs = []
    for tr, va in cv.split(ds.X_train, ds.y_train):
        m = make_model(iterations=best_it)
        m.fit(Pool(ds.X_train.iloc[tr], ds.y_train[tr], cat_features=cats))
        aucs.append(roc_auc_score(ds.y_train[va], m.predict_proba(Pool(ds.X_train.iloc[va], cat_features=cats))[:, 1]))
    return {"proba_test": proba, "thr_y": y_val, "thr_proba": val_proba, "train_seconds": t1 - t0,
            "predict_seconds": t2 - t1, "cv_roc_auc_mean": float(np.mean(aucs)), "cv_roc_auc_std": float(np.std(aucs)),
            "model": model, "best_iteration": best_it, "evals": model.get_evals_result()}


def ordered_ts_demo(ds, level="student", col="job", a=1.0, n_show=400):
    """Formula f1 computed directly: running ordered target statistic for one category."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    perm = rng.permutation(len(ds.X_train))
    x = ds.X_train[col].to_numpy()[perm]
    y = ds.y_train[perm]
    p = float(ds.y_train.mean())
    rows = np.where(x == level)[0][:n_show]
    hits = np.cumsum(np.r_[0, y[rows][:-1]])          # sum of y over *earlier* rows with this level
    counts = np.arange(len(rows))                     # number of earlier rows with this level
    ts = (hits + a * p) / (counts + a)
    true_rate = float(ds.y_train[ds.X_train[col].to_numpy() == level].mean())
    return rows, ts, true_rate, p


META = {
    "model_id": MID, "display_name": "CatBoost", "family": "gradient_boosting", "task": "supervised_classification",
    "one_liner": "Builds trees in sequence, each correcting the previous one's mistakes.",
    "plain_explanation": (
        "Boosting builds many small trees in sequence. The first tree makes a rough guess. The second tree is "
        "trained not on the original answer but on the first tree's mistakes. The third corrects what remains, "
        "and so on. The final prediction is the sum of all of them, each scaled down by a small learning rate so no "
        "single tree dominates. CatBoost's distinctive part is how it handles categorical columns like job or month: "
        "instead of one-hot encoding, it replaces each category with a running average of the target — computed only "
        "from rows seen earlier, which is what stops it leaking the answer."),
    "year": 2017, "reference": "Prokhorenkova et al., CatBoost: unbiased boosting with categorical features, NeurIPS 2018",
    "workflow": [
        {"id": "n1", "label": "Encode categoricals", "detail": "Ordered target statistics. Only earlier rows contribute, which prevents target leakage.", "formula_ref": "f1"},
        {"id": "n2", "label": "Initialise", "detail": "Start from the log-odds of the base rate — the best constant guess.", "formula_ref": "f2"},
        {"id": "n3", "label": "Compute residuals", "detail": "How wrong are we on each row right now?", "formula_ref": "f3"},
        {"id": "n4", "label": "Fit a tree to the residuals", "detail": "A shallow symmetric (oblivious) tree predicting the current errors: every node at the same depth asks the same question.", "formula_ref": None},
        {"id": "n5", "label": "Update with shrinkage", "detail": "Add the new tree, scaled by the learning rate.", "formula_ref": "f4"},
        {"id": "n6", "label": "Repeat or early-stop", "detail": "Loop to n3 until validation AUC stops improving for 50 rounds.", "formula_ref": "f5"},
    ],
    "edges": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"], ["n4", "n5"], ["n5", "n3"], ["n5", "n6"]],
    "formulas": [
        {"id": "f1", "name": "Ordered target statistic",
         "latex": r"\hat{x}_k = \frac{\sum_{j=1}^{k-1}\mathbb{1}[x_j = x_k]\,y_j + a\,p}{\sum_{j=1}^{k-1}\mathbb{1}[x_j = x_k] + a}",
         "caption": "Replaces a category with the average target of earlier rows sharing that category. The prior p and weight a smooth rare categories. Using only j < k is what makes it leak-free — a naive target encoding using all rows would leak the answer.",
         "glossary": [{"sym": "a", "means": "smoothing weight"}, {"sym": "p", "means": "global base rate (0.1127 here)"}, {"sym": r"\mathbb{1}[\cdot]", "means": "1 if true, else 0"}]},
        {"id": "f2", "name": "Initial prediction", "latex": r"F_0(x) = \log\frac{p}{1-p}",
         "caption": "The log-odds of the base rate. Here log(0.1127/0.8873) ≈ −2.06 — the best possible constant guess.", "glossary": []},
        {"id": "f3", "name": "Residual (pseudo-gradient)", "latex": r"r_i = y_i - \sigma\big(F_{m-1}(x_i)\big), \qquad \sigma(z) = \frac{1}{1+e^{-z}}",
         "caption": "How wrong the current ensemble is on row i. The next tree is trained to predict this, not y.",
         "glossary": [{"sym": r"\sigma", "means": "logistic sigmoid"}]},
        {"id": "f4", "name": "Boosting update", "latex": r"F_m(x) = F_{m-1}(x) + \eta\, h_m(x)",
         "caption": "Each tree is added scaled by the learning rate η. Small η means slow, stable learning and needs more trees.",
         "glossary": [{"sym": r"\eta", "means": "learning rate (0.05)"}, {"sym": "h_m", "means": "the m-th tree"}]},
        {"id": "f5", "name": "Loss (log loss)",
         "latex": r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log \sigma(F(x_i)) + (1-y_i)\log\big(1-\sigma(F(x_i))\big)\Big]",
         "caption": "What each tree reduces. The residual in f3 is exactly the negative gradient of this loss.", "glossary": []},
    ],
    "assumptions": [
        {"text": "Enough rows per categorical level", "status": "warn", "evidence": "job=unknown has 330 rows (0.80%), education=illiterate 18; the prior term a·p in f1 smooths them."},
        {"text": "No scaling required", "status": "ok", "evidence": "Trees are scale-invariant."},
        {"text": "Independent rows", "status": "warn", "evidence": "Data is time-ordered May 2008–Nov 2010; a random split ignores this."},
    ],
}


def figures(out, out_w, ds):
    figs = []
    ev = out["evals"]
    lt, lv = ev["learn"]["Logloss"], ev["validation"]["Logloss"]
    auc_v = ev["validation"]["AUC"]
    it = list(range(1, len(lt) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=it, y=lt, name="train log loss", line=dict(color=MUTED)))
    fig.add_trace(go.Scatter(x=it, y=lv, name="validation log loss", line=dict(color=ACCENT, width=2.5)))
    fig.add_trace(go.Scatter(x=it, y=auc_v, name="validation AUC", yaxis="y2", line=dict(color=AMBER, dash="dot")))
    fig.add_vline(x=out["best_iteration"], line_dash="dot", line_color=ACCENT,
                  annotation_text=f"best iteration {out['best_iteration']}")
    fig.update_layout(title="Training curve (early stopping on validation AUC, patience 50)", xaxis_title="iteration",
                      yaxis_title="log loss (class-weighted)", yaxis2=dict(title="AUC", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.2))
    write_figure(MID, "loss_curve.json", fig)
    figs.append({"file": "loss_curve.json", "title": "Training and validation loss", "type": "line",
                 "caption": f"Validation AUC peaked at iteration {out['best_iteration']} of 1,000; training stopped "
                            f"{EARLY_STOP} rounds later. With duration it ran to {out_w['best_iteration']}."})

    model = out["model"]
    imp = model.get_feature_importance(type="PredictionValuesChange")
    names = model.feature_names_
    order = np.argsort(imp)
    fig = go.Figure(go.Bar(x=imp[order], y=[names[i] for i in order], orientation="h", marker_color=ACCENT,
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_layout(title="Feature importance (PredictionValuesChange, sums to 100)", margin=dict(l=120))
    write_figure(MID, "feature_importance.json", fig)
    top = names[order[-1]]
    figs.append({"file": "feature_importance.json", "title": "Feature importance", "type": "bar",
                 "caption": f"{top} leads with {imp[order[-1]]:.1f}% of the total prediction change. "
                            f"day_of_week: {imp[names.index('day_of_week')]:.1f}%."})

    # One extracted tree: CatBoost trees are oblivious — the same split at every node of a level.
    pool = Pool(ds.X_train.head(200), cat_features=ds.categorical)
    splits = model._get_tree_splits(0, pool)
    n_leaves = int(model.get_tree_leaf_counts()[0])
    leaf_vals = np.asarray(model.get_leaf_values()[:n_leaves], dtype=float)
    fig = go.Figure(go.Bar(x=list(range(len(leaf_vals))), y=leaf_vals,
                           marker_color=[ACCENT if v > 0 else MUTED for v in leaf_vals],
                           hovertemplate="leaf %{x}: %{y:.4f}<extra></extra>"))
    fig.update_layout(title=f"Tree #1 of the ensemble: {len(splits)} questions, {n_leaves} leaf values", xaxis_title="leaf (binary code of the answers)",
                      yaxis_title="contribution to log-odds (before ×η)",
                      annotations=[dict(x=0, y=1.0 + 0.07 * (len(splits) - k), xref="paper", yref="paper", showarrow=False,
                                        xanchor="left", text=f"level {k + 1}: {s}", font=dict(size=10))
                                   for k, s in enumerate(splits)], margin=dict(t=60 + 18 * len(splits)))
    write_figure(MID, "first_tree.json", fig)
    figs.append({"file": "first_tree.json", "title": "One tree from the ensemble", "type": "bar",
                 "caption": f"CatBoost grows oblivious trees: every node at a level asks the same question, so this tree "
                            f"is just {len(splits)} questions and 2^{len(splits)} = {n_leaves} leaves (depth 6 is a maximum, not a requirement). Splits on categorical features use their ordered target "
                            "statistic (shown as a ctr border)."})

    y, p = ds.y_test, out["proba_test"]
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="perfect calibration", line=dict(color=MUTED, dash="dot")))
    fig.add_trace(go.Scatter(x=mean_pred, y=frac, name="CatBoost (test)", mode="lines+markers", line=dict(color=ACCENT, width=2.5)))
    fig.update_layout(title="Calibration (10 quantile bins, test set)", xaxis_title="mean predicted probability",
                      yaxis_title="observed subscription rate", legend=dict(orientation="h", y=-0.2))
    write_figure(MID, "calibration.json", fig)
    figs.append({"file": "calibration.json", "title": "Calibration curve", "type": "line",
                 "caption": f"Balanced class weights make the model over-confident: its mean prediction is {p.mean():.2f} "
                            f"while the true rate is {y.mean():.2f}. Ranking (AUC) is unaffected; raw probabilities are not "
                            "real chances."})

    rows, ts, true_rate, prior = ordered_ts_demo(ds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(ts) + 1)), y=ts, name="ordered target statistic", line=dict(color=ACCENT)))
    fig.add_hline(y=true_rate, line_dash="dash", line_color=AMBER, annotation_text=f"true train rate {true_rate:.3f}")
    fig.add_hline(y=prior, line_dash="dot", line_color=MUTED, annotation_text=f"prior p = {prior:.4f}",
                  annotation_position="bottom right")
    fig.update_layout(title="Formula f1 in action: job = student, one random permutation", xaxis_title="k-th student row in the permutation",
                      yaxis_title="encoded value x̂ₖ")
    write_figure(MID, "ordered_ts.json", fig)
    figs.append({"file": "ordered_ts.json", "title": "Ordered target statistic", "type": "line",
                 "caption": "Each student row is encoded using only the students before it. The first is pure prior (0.113); "
                            "the estimate converges toward the true student rate. No row ever sees its own label."})
    return figs


def main():
    print("CatBoost: two runs with early stopping + 5-fold CV…")
    outs = run_both(MID, "native", fit_predict,
                    {"resampling": "auto_class_weights='Balanced'", "validation": f"{VAL_SIZE:.0%} of train held out for early stopping and threshold"})
    ds, out = outs["without_duration"]
    _, out_w = outs["with_duration"]
    d = model_dir(MID)
    figs = figures(out, out_w, ds)
    write_json(d / "meta.json", {**META, "figures": figs})
    write_json(d / "config.json", {
        "estimator": "catboost.CatBoostClassifier",
        **params_block([
            ("iterations", 1000, "Upper bound only; early stopping decides the real count."),
            ("learning_rate", 0.05, "Small steps (η in f4): more trees, smoother fit, less overfitting."),
            ("depth", 6, "CatBoost's default; oblivious trees of depth 6 capture up to 6-way interactions."),
            ("early_stopping_rounds", EARLY_STOP, f"Stop when validation AUC has not improved for 50 rounds. Stopped at {out['best_iteration']} trees."),
            ("eval_metric", "AUC", "We rank clients; AUC measures ranking quality directly."),
            ("auto_class_weights", "Balanced", "Only 11.27% subscribe; reweighting stops the loss being dominated by 'no'."),
            ("cat_features", ds.categorical, "Passed natively — no one-hot. CatBoost encodes them with ordered target statistics (f1)."),
            ("random_seed", C.RANDOM_STATE, "Fixes the permutation used for ordered statistics, so results are reproducible."),
        ]),
    })

    # SHAP via CatBoost's built-in exact TreeSHAP (handles native categoricals directly)
    n = min(SAMPLE_CAPS[MID], len(ds.X_test))
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = np.sort(rng.choice(len(ds.X_test), n, replace=False))
    X_raw = ds.X_test.iloc[rows].reset_index(drop=True)
    sv = out["model"].get_feature_importance(Pool(X_raw, cat_features=ds.categorical), type="ShapValues")
    payload = build_payload(MID, "CatBoost", "TreeExplainer (CatBoost native TreeSHAP)", sv[:, :-1], X_raw,
                            out["proba_test"][rows], ds.y_test[rows], float(sv[0, -1]), "log-odds", ds.test_index[rows])
    write_explain(MID, payload)

    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    wo, w = m["runs"]["without_duration"], m["runs"]["with_duration"]
    vs = None
    dt_pred = C.RESULTS_MODELS / "decision_tree" / "predictions.json"
    if dt_pred.exists():
        dtp = json.loads(dt_pred.read_text())
        cbp = json.loads((d / "predictions.json").read_text())
        dtm = json.loads((C.RESULTS_MODELS / "decision_tree" / "metrics.json").read_text())["runs"]["without_duration"]
        vs = (mcnemar(cbp["y_true"], cbp["without_duration"]["pred"], dtp["without_duration"]["pred"]), dtm)
    findings = [
        {"type": "result", "text": f"Without duration: ROC-AUC {wo['roc_auc']:.4f}, PR-AUC {wo['pr_auc']:.4f}, F1 {wo['f1']:.4f}; "
                                   f"5-fold CV AUC {wo['cv_roc_auc_mean']:.4f} ± {wo['cv_roc_auc_std']:.4f}. "
                                   "The plan predicted 0.80–0.82 (HistGradientBoosting measured 0.8136)."},
        {"type": "result", "text": f"Leak inflation: +{m['leak_delta']['roc_auc']:.4f} AUC, +{m['leak_delta']['pr_auc']:.4f} PR-AUC. "
                                   f"With duration early stopping ran to {out_w['best_iteration']} trees instead of {out['best_iteration']}."},
    ]
    if vs:
        mc, dtm = vs
        sig = "is" if mc["p_value"] < 0.05 else "is not"
        findings.append({"type": "result", "text": f"Against the decision tree: PR-AUC {wo['pr_auc']:.4f} vs {dtm['pr_auc']:.4f} "
                                                   f"(margin {wo['pr_auc'] - dtm['pr_auc']:+.4f}). McNemar on test predictions: "
                                                   f"p = {mc['p_value']:.3g} — the difference in errors {sig} statistically significant at 0.05."})
    findings += [
        {"type": "limitation", "text": f"Balanced weights leave probabilities over-confident (mean prediction {out['proba_test'].mean():.2f} vs base rate 0.11, Brier {wo['brier']:.4f}). Use the scores for ranking, or recalibrate before reading them as chances."},
        {"type": "limitation", "text": "Ordered target statistics depend on a random permutation; rare levels (education=illiterate, 18 rows) get noisy encodings dominated by the prior."},
        {"type": "limitation", "text": "Gains over a single tree come with ~100× more trees and much less transparency; the extracted tree is one of hundreds."},
    ]
    write_json(d / "findings.json", {
        "headline": f"CatBoost reaches AUC {wo['roc_auc']:.3f} without the leak — the strongest legitimate ranking so far, but by a modest margin.",
        "findings": findings,
        "comparison_note": "Same family of base learner as the decision tree, but hundreds of them added in sequence, and categorical features encoded by their response rate rather than split one level at a time.",
        "viva_answer": "Q: Why does CatBoost handle categorical features better than one-hot encoding? — One-hot turns job into 12 sparse columns, and a tree can then only ask 'is this job admin, yes or no?' one level at a time. Ordered target statistics replace the category with a number carrying the response rate directly, so a single split can separate high-responding jobs from low-responding ones. The 'ordered' part matters: averaging over all rows including the current one would leak the target, so CatBoost uses only rows seen earlier in a random permutation.",
    })
    validate_model_artifacts(MID)
    print("catboost artifacts valid")


if __name__ == "__main__":
    main()
