"""Agent-01 — Decision Tree (§7.1).   python -m src.models.decision_tree"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from scipy.stats import entropy as scipy_entropy
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .. import config as C
from ..contract import model_dir, validate_model_artifacts, write_figure, write_json
from ..explain import SAMPLE_CAPS, aggregate_onehot, build_payload, write_explain
from ..preprocessing import make_onehot_preprocessor, onehot_source_map
from ._supervised import params_block, run_both

MID = "decision_tree"
DEPTHS = list(range(1, 16))
MIN_LEAF = 50
ACCENT, MUTED, AMBER = "#4FB3A0", "#6B7E8F", "#D4A373"


def make_pipe(ds, depth):
    return Pipeline([
        ("prep", make_onehot_preprocessor(ds.categorical, ds.numeric, scale=False)),
        ("tree", DecisionTreeClassifier(criterion="gini", max_depth=depth, min_samples_leaf=MIN_LEAF,
                                        class_weight="balanced", random_state=C.RANDOM_STATE)),
    ])


def fit_predict(ds):
    cv = StratifiedKFold(C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    curve = []
    for d in DEPTHS:
        r = cross_validate(make_pipe(ds, d), ds.X_train, ds.y_train, cv=cv, n_jobs=-1,
                           scoring=["roc_auc", "accuracy"], return_train_score=True)
        curve.append({"depth": d, **{k: float(np.mean(v)) for k, v in r.items() if k.startswith(("test_", "train_"))},
                      "cv_auc_std": float(np.std(r["test_roc_auc"]))})
    best = max(curve, key=lambda c: c["test_roc_auc"])
    depth = best["depth"]
    oof = cross_val_predict(make_pipe(ds, depth), ds.X_train, ds.y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    pipe = make_pipe(ds, depth)
    t0 = time.perf_counter(); pipe.fit(ds.X_train, ds.y_train); t1 = time.perf_counter()
    proba = pipe.predict_proba(ds.X_test)[:, 1]; t2 = time.perf_counter()
    return {"proba_test": proba, "thr_y": ds.y_train, "thr_proba": oof, "train_seconds": t1 - t0,
            "predict_seconds": t2 - t1, "cv_roc_auc_mean": best["test_roc_auc"],
            "cv_roc_auc_std": best["cv_auc_std"], "model": pipe, "depth": depth, "curve": curve}


# ------------------------------------------------------------------ tree export

def readable_split(col, thr, cats):
    for c in cats:
        if col.startswith(c + "_"):
            return f"{c} = {col[len(c) + 1:]}", f"{c} ≠ {col[len(c) + 1:]}"
    t = f"{thr:,.1f}" if abs(thr) >= 100 else f"{thr:.3g}"
    return f"{col} ≤ {t}", f"{col} > {t}"


def export_tree(pipe, ds, max_depth=3):
    prep, tree = pipe.named_steps["prep"], pipe.named_steps["tree"]
    names = list(prep.get_feature_names_out())
    Xt = prep.transform(ds.X_train)
    paths = tree.decision_path(Xt)
    yes = ds.y_train.astype(bool)
    t = tree.tree_
    nodes = []

    def walk(i, depth, parent, edge):
        idx = paths[:, i].nonzero()[0]
        n_yes = int(yes[idx].sum())
        node = {"id": int(i), "parent": parent, "edge": edge, "depth": depth, "samples": int(len(idx)),
                "yes": n_yes, "no": int(len(idx) - n_yes), "rate": n_yes / max(len(idx), 1),
                "gini_weighted": float(t.impurity[i]), "leaf": bool(t.children_left[i] == -1),
                "truncated": False}
        if not node["leaf"]:
            left_lbl, right_lbl = readable_split(names[t.feature[i]], t.threshold[i], ds.categorical)
            # one-hot columns: value ≤ 0.5 means "not this level" → the left branch is "≠"
            is_cat = left_lbl.count("=") == 1 and "≤" not in left_lbl
            node["split"] = left_lbl
            if depth >= max_depth:
                node["truncated"] = True
            else:
                l_edge, r_edge = (right_lbl, left_lbl) if is_cat else (left_lbl, right_lbl)
                nodes.append(node)
                walk(t.children_left[i], depth + 1, int(i), l_edge)
                walk(t.children_right[i], depth + 1, int(i), r_edge)
                return
        nodes.append(node)

    walk(0, 0, None, None)
    nodes.sort(key=lambda n: (n["depth"], n["id"]))
    return {"type": "tree", "max_depth_shown": max_depth, "model_depth": int(tree.get_depth()),
            "n_leaves": int(tree.get_n_leaves()), "nodes": nodes,
            "note": "Counts are real training rows reaching each node; the split criterion used "
                    "class-balanced weights."}


# ------------------------------------------------------------------ hand verification

def H(counts):
    p = np.asarray(counts, float) / np.sum(counts)
    p = p[p > 0]  # 0·log2(0) = 0 by convention (lim p→0⁺ p·log p = 0)
    return float(-(p * np.log2(p)).sum())


def G(counts):
    p = np.asarray(counts, float) / np.sum(counts)
    return float(1 - (p ** 2).sum())


def multiway_scores(x: pd.Series, y: np.ndarray):
    tot = [int((y == 1).sum()), int((y == 0).sum())]
    N = len(y)
    levels = []
    for lvl, grp in pd.Series(y).groupby(x.to_numpy()):
        levels.append({"level": str(lvl), "n": int(len(grp)), "yes": int(grp.sum()), "no": int(len(grp) - grp.sum())})
    after_h = sum(l["n"] / N * H([l["yes"], l["no"]]) for l in levels)
    after_g = sum(l["n"] / N * G([l["yes"], l["no"]]) for l in levels)
    split_info = H([l["n"] for l in levels])
    ig = H(tot) - after_h
    return {"levels": levels, "H_S": H(tot), "weighted_H_after": after_h, "info_gain": ig,
            "split_info": split_info, "gain_ratio": ig / split_info if split_info > 0 else None,
            "gini_S": G(tot), "weighted_gini_after": after_g, "gini_gain": G(tot) - after_g}


def hand_verification(ds):
    y = ds.y_train
    x = ds.X_train["poutcome"]
    s = multiway_scores(x, y)
    N, ny = len(y), int(y.sum())
    nn = N - ny

    def term(k, n):
        return f"({k:,}/{n:,})·log₂({k:,}/{n:,})"

    steps = [
        {"label": "Root entropy H(S)",
         "expr": f"H(S) = −{term(ny, N)} − {term(nn, N)}", "value": s["H_S"], "formula_ref": "f1"},
    ]
    for l in s["levels"]:
        steps.append({"label": f"H(S | poutcome={l['level']})",
                      "expr": f"−{term(l['yes'], l['n'])} − {term(l['no'], l['n'])}", "value": H([l["yes"], l["no"]]),
                      "formula_ref": "f1"})
    steps += [
        {"label": "Weighted entropy after the split",
         "expr": " + ".join(f"({l['n']:,}/{N:,})·{H([l['yes'], l['no']]):.4f}" for l in s["levels"]),
         "value": s["weighted_H_after"], "formula_ref": "f2"},
        {"label": "Information gain (ID3)", "expr": f"{s['H_S']:.4f} − {s['weighted_H_after']:.4f}",
         "value": s["info_gain"], "formula_ref": "f2"},
        {"label": "Split information", "expr": "−Σ (|Sₖ|/|S|)·log₂(|Sₖ|/|S|) over " +
         ", ".join(f"{l['n']:,}" for l in s["levels"]), "value": s["split_info"], "formula_ref": "f3"},
        {"label": "Gain ratio (C4.5)", "expr": f"{s['info_gain']:.4f} / {s['split_info']:.4f}",
         "value": s["gain_ratio"], "formula_ref": "f3"},
        {"label": "Root Gini G(S)", "expr": f"1 − ({ny:,}/{N:,})² − ({nn:,}/{N:,})²", "value": s["gini_S"],
         "formula_ref": "f4"},
        {"label": "Gini gain (3-way)", "expr": f"{s['gini_S']:.4f} − " + " − ".join(
            f"({l['n']:,}/{N:,})·{G([l['yes'], l['no']]):.4f}" for l in s["levels"]),
         "value": s["gini_gain"], "formula_ref": "f5"},
    ]

    # Code checks: scipy for the multiway entropy; sklearn stumps for the binary CART split.
    code = {"scipy_H_S": float(scipy_entropy([ny, nn], base=2)),
            "scipy_weighted_H_after": float(sum(l["n"] / N * scipy_entropy([l["yes"], l["no"]], base=2) for l in s["levels"])),
            "scipy_split_info": float(scipy_entropy([l["n"] for l in s["levels"]], base=2))}
    dummies = pd.get_dummies(x, prefix="poutcome").astype(int)
    binary = {}
    for crit in ("entropy", "gini"):
        stump = DecisionTreeClassifier(criterion=crit, max_depth=1, random_state=C.RANDOM_STATE).fit(dummies, y)
        t = stump.tree_
        n0, nl, nr = t.n_node_samples[0], t.n_node_samples[1], t.n_node_samples[2]
        gain = t.impurity[0] - (nl / n0) * t.impurity[1] - (nr / n0) * t.impurity[2]
        # Reproduce the same binary split by hand
        col = dummies.columns[t.feature[0]]
        left = dummies[col].to_numpy() <= 0.5
        f = H if crit == "entropy" else G
        hand_children = [f([int(y[m].sum()), int((~y[m].astype(bool)).sum())]) for m in (left, ~left)]
        hand_gain = f([ny, nn]) - left.mean() * hand_children[0] - (~left).mean() * hand_children[1]
        binary[crit] = {"split": f"{col.replace('poutcome_', 'poutcome = ')} vs rest",
                        "sklearn_root_impurity": float(t.impurity[0]),
                        "sklearn_children_impurity": [float(t.impurity[1]), float(t.impurity[2])],
                        "sklearn_children_samples": [int(nl), int(nr)],
                        "sklearn_gain": float(gain), "hand_gain": float(hand_gain),
                        "match": bool(abs(gain - hand_gain) < 1e-9)}

    # Every categorical feature: shows why gain ratio exists (high-cardinality job/education/month)
    table = []
    for c in ds.categorical:
        m = multiway_scores(ds.X_train[c], y)
        table.append({"feature": c, "levels": len(m["levels"]), "info_gain": m["info_gain"],
                      "split_info": m["split_info"], "gain_ratio": m["gain_ratio"], "gini_gain": m["gini_gain"]})
    table.sort(key=lambda r: -r["info_gain"])
    avg_ig = float(np.mean([r["info_gain"] for r in table]))
    survivors = [r for r in table if r["info_gain"] >= avg_ig]
    c45_pick = max(survivors, key=lambda r: r["gain_ratio"])["feature"]

    return {
        "type": "hand_verification",
        "feature": "poutcome",
        "data": f"training split, {N:,} rows, unweighted counts",
        "root": {"n": N, "yes": ny, "no": nn},
        "levels": s["levels"],
        "steps": steps,
        "code": code,
        "checks": [
            {"what": "H(S)", "hand": s["H_S"], "code": code["scipy_H_S"], "source": "scipy.stats.entropy"},
            {"what": "Weighted H after", "hand": s["weighted_H_after"], "code": code["scipy_weighted_H_after"], "source": "scipy.stats.entropy"},
            {"what": "Split information", "hand": s["split_info"], "code": code["scipy_split_info"], "source": "scipy.stats.entropy"},
            {"what": f"Binary entropy gain ({binary['entropy']['split']})", "hand": binary["entropy"]["hand_gain"],
             "code": binary["entropy"]["sklearn_gain"], "source": "sklearn tree_.impurity"},
            {"what": f"Binary Gini gain ({binary['gini']['split']})", "hand": binary["gini"]["hand_gain"],
             "code": binary["gini"]["sklearn_gain"], "source": "sklearn tree_.impurity"},
        ],
        "binary_cart": binary,
        "all_features": table,
        "c45": {"average_info_gain": avg_ig, "survivors": [r["feature"] for r in survivors], "pick": c45_pick,
                "id3_pick": table[0]["feature"],
                "gini_pick": max(table, key=lambda r: r["gini_gain"])["feature"]},
        "note": "sklearn's CART only makes binary splits, so the 3-way ID3/C4.5 numbers are checked "
                "against scipy, and the binary split CART actually makes is checked against sklearn.",
    }


# ------------------------------------------------------------------ meta (§7.1, verbatim)

META = {
    "model_id": MID, "display_name": "Decision Tree", "family": "interpretable_baseline",
    "task": "supervised_classification",
    "one_liner": "Plays twenty questions: asks the most informative question first, then repeats on each group.",
    "plain_explanation": (
        "A decision tree plays twenty questions. At each step it picks the question that most reduces "
        "uncertainty about the answer, splits the data on it, and repeats on each group until a group is "
        "pure enough or the tree hits a depth limit. To predict, you follow one path from root to leaf. "
        "The entire subject is one question: which question do you ask first? Every variant — ID3, C4.5, "
        "CART — is a different answer to that."),
    "year": 1984, "reference": "Breiman et al., Classification and Regression Trees (1984); Quinlan, ID3 (1986), C4.5 (1993)",
    "workflow": [
        {"id": "n1", "label": "Measure impurity at the node", "detail": "How mixed are the classes here? Zero means pure.", "formula_ref": "f1,f4"},
        {"id": "n2", "label": "Score every candidate split", "detail": "For each feature and threshold, compute the weighted impurity after splitting.", "formula_ref": "f2,f5"},
        {"id": "n3", "label": "Correct for cardinality", "detail": "A feature with many values looks good for the wrong reason. Dividing by split information penalises fragmentation.", "formula_ref": "f3"},
        {"id": "n4", "label": "Choose the best split", "detail": "Take the feature with the largest reduction.", "formula_ref": None},
        {"id": "n5", "label": "Partition and recurse", "detail": "Send rows down each branch and repeat.", "formula_ref": None},
        {"id": "n6", "label": "Stop", "detail": "Pure node, no features left, depth limit, or minimum samples reached.", "formula_ref": None},
        {"id": "n7", "label": "Prune", "detail": "Remove branches that do not improve validation performance. Here: max_depth chosen by 5-fold CV and min_samples_leaf=50 act as pre-pruning.", "formula_ref": None},
    ],
    "edges": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"], ["n4", "n5"], ["n5", "n1"], ["n5", "n6"], ["n6", "n7"]],
    "formulas": [
        {"id": "f1", "name": "Entropy", "latex": r"H(S) = -\sum_{i=1}^{k} p_i \log_2 p_i",
         "caption": "The average number of yes/no questions needed to identify a class. Zero means you already know; 1 bit means a balanced coin flip. Convention: 0·log₂0 = 0, justified by lim p→0⁺ p log p = 0.",
         "glossary": [{"sym": "p_i", "means": "proportion of class i"}, {"sym": "k", "means": "number of classes"}]},
        {"id": "f2", "name": "Information gain (ID3)", "latex": r"IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)",
         "caption": "Uncertainty before the split minus the weighted average after. Weighted, because a branch holding one row should not count as much as one holding a thousand. IG ≥ 0 always — a negative value means an arithmetic error.",
         "glossary": [{"sym": "S_v", "means": "rows where feature A takes value v"}]},
        {"id": "f3", "name": "Split information and gain ratio (C4.5)",
         "latex": r"\text{SplitInfo}(A) = -\sum_{k} \frac{|S_k|}{|S|} \log_2 \frac{|S_k|}{|S|}, \qquad \text{GainRatio}(A) = \frac{IG(A)}{\text{SplitInfo}(A)}",
         "caption": "Split information is the entropy of the branch-size distribution, ignoring class labels entirely — it measures fragmentation. An ID column would score maximum information gain and be useless; dividing by split information fixes that. Fails when a split is very unbalanced (SplitInfo → 0), so real C4.5 first discards features below the average information gain.",
         "glossary": [{"sym": "S_k", "means": "rows sent to branch k"}]},
        {"id": "f4", "name": "Gini impurity (CART)", "latex": r"G(S) = 1 - \sum_{i=1}^{k} p_i^2",
         "caption": "The probability of misclassifying a random row if you guessed its class by drawing from the node's distribution. Range [0, 1−1/k]; for binary, [0, 0.5]. Never compare a Gini value to an entropy value — different scales.",
         "glossary": [{"sym": "p_i", "means": "proportion of class i"}]},
        {"id": "f5", "name": "Gini gain", "latex": r"\Delta G = G(S) - \sum_{v} \frac{|S_v|}{|S|} G(S_v)",
         "caption": "The CART criterion actually used by this model (with class-balanced weights).", "glossary": []},
    ],
    "assumptions": [
        {"text": "No distributional assumption needed", "status": "ok", "evidence": "Trees only compare values against thresholds."},
        {"text": "Features not required to be scaled", "status": "ok", "evidence": "Unlike kNN or neural nets; we pass raw values."},
        {"text": "Low-cardinality features preferred for ID3", "status": "warn", "evidence": "job has 12 levels, education 8, month 10 — use gain ratio or Gini (see the hand-verification table)."},
        {"text": "Axis-parallel boundaries only", "status": "warn", "evidence": "Cannot express diagonal boundaries without deep trees; the macro features move together, which a single split cannot exploit."},
    ],
}


def figures(dt_out, dt_out_w, ds):
    figs = []
    curve, curve_w = dt_out["curve"], dt_out_w["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DEPTHS, y=[c["train_roc_auc"] for c in curve], name="train AUC", line=dict(color=MUTED, dash="dot")))
    fig.add_trace(go.Scatter(x=DEPTHS, y=[c["test_roc_auc"] for c in curve], name="CV AUC", line=dict(color=ACCENT, width=2.5),
                             error_y=dict(type="data", array=[c["cv_auc_std"] for c in curve], visible=True, thickness=1)))
    fig.add_trace(go.Scatter(x=DEPTHS, y=[c["test_roc_auc"] for c in curve_w], name="CV AUC with duration (leaky)",
                             line=dict(color=AMBER, dash="dash")))
    fig.add_vline(x=dt_out["depth"], line_dash="dot", line_color=ACCENT,
                  annotation_text=f"chosen depth = {dt_out['depth']}")
    fig.update_layout(title="AUC vs max_depth (5-fold CV on train)", xaxis_title="max_depth", yaxis_title="ROC-AUC",
                      legend=dict(orientation="h", y=-0.2))
    write_figure(MID, "depth_curve.json", fig)
    gap = curve[-1]["train_roc_auc"] - curve[-1]["test_roc_auc"]
    figs.append({"file": "depth_curve.json", "title": "Depth vs AUC", "type": "line",
                 "caption": f"CV AUC peaks at depth {dt_out['depth']} ({dt_out['cv_roc_auc_mean']:.4f}). Depth 3 already reaches "
                            f"{curve[2]['test_roc_auc']:.4f}; at depth 15 the train–CV gap is {gap:.3f} — the overfitting point."})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DEPTHS, y=[c["train_accuracy"] for c in curve], name="train accuracy", line=dict(color=MUTED, dash="dot")))
    fig.add_trace(go.Scatter(x=DEPTHS, y=[c["test_accuracy"] for c in curve], name="CV accuracy", line=dict(color=ACCENT)))
    fig.add_hline(y=C.MAJORITY_ACCURACY, line_dash="dash", line_color=AMBER, annotation_text="majority baseline 0.8873")
    fig.update_layout(title="Accuracy vs max_depth (balanced class weights, threshold 0.5)", xaxis_title="max_depth",
                      yaxis_title="accuracy", legend=dict(orientation="h", y=-0.2))
    write_figure(MID, "depth_accuracy.json", fig)
    figs.append({"file": "depth_accuracy.json", "title": "Depth vs accuracy", "type": "line",
                 "caption": "With balanced class weights every depth sits below the 'always no' baseline at threshold 0.5 — "
                            "the tree is trading accuracy for recall on purpose. Accuracy is the wrong yardstick here."})

    pipe = dt_out["model"]
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    src = onehot_source_map(names, ds.categorical)
    imp = pd.Series(pipe.named_steps["tree"].feature_importances_, index=names)
    agg = imp.groupby(imp.index.map(src)).sum().sort_values()
    fig = go.Figure(go.Bar(x=agg.values, y=agg.index, orientation="h", marker_color=ACCENT,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Feature importance (total Gini reduction, one-hot columns summed)", xaxis_title="importance",
                      margin=dict(l=120))
    write_figure(MID, "feature_importance.json", fig)
    zero = [f for f, v in agg.items() if v == 0]
    figs.append({"file": "feature_importance.json", "title": "Feature importance", "type": "bar",
                 "caption": f"Top: {agg.index[-1]} ({agg.iloc[-1]:.2f}). "
                            + (f"Never used in any split: {', '.join(zero)}." if zero else "Every feature is used at least once.")})

    write_json(model_dir(MID) / "figures" / "tree.json", export_tree(pipe, ds))
    figs.insert(0, {"file": "tree.json", "title": "The fitted tree (top 3 levels)", "type": "tree",
                    "caption": f"The real model has depth {pipe.named_steps['tree'].get_depth()} and "
                               f"{pipe.named_steps['tree'].get_n_leaves()} leaves; only the first three levels are drawn. "
                               "Each node shows actual training rows and their subscription rate."})
    return figs, agg


def depth_sentence(d_wo, d_w):
    if d_w > d_wo:
        return f"With duration the CV-best depth rises from {d_wo} to {d_w} — the tree spends extra levels carving call-length thresholds."
    if d_w < d_wo:
        return f"With duration the CV-best depth falls from {d_wo} to {d_w}: one sharp feature makes extra levels unnecessary."
    return f"The CV-best depth is {d_wo} in both runs: the leak changes which splits are made, not how deep the tree should be."


def main():
    print("Decision tree: tuning depth 1–15 by 5-fold CV for each run…")
    outs = run_both(MID, "onehot", fit_predict,
                    {"resampling": "class_weight='balanced' (reweighting, no resampling)", "tuning": "max_depth 1–15 by stratified 5-fold CV AUC"})
    ds, out = outs["without_duration"]
    _, out_w = outs["with_duration"]
    d = model_dir(MID)

    figs, agg = figures(out, out_w, ds)
    hv = hand_verification(ds)
    write_json(d / "hand_verification.json", hv)

    meta = {**META, "figures": figs, "extras": [{"kind": "hand_verification", "file": "hand_verification.json",
                                                  "title": "Hand verification: poutcome at the root"}]}
    write_json(d / "meta.json", meta)

    write_json(d / "config.json", {
        "estimator": "sklearn.tree.DecisionTreeClassifier",
        **params_block([
            ("criterion", "gini", "CART's criterion; cheaper than entropy (no logarithms) and rarely disagrees — both are shown in the hand-verification panel."),
            ("max_depth", out["depth"], f"Chosen from 1–15 by stratified 5-fold CV AUC on the training split (best CV AUC {out['cv_roc_auc_mean']:.4f}). With duration the best depth was {out_w['depth']}."),
            ("min_samples_leaf", MIN_LEAF, "Pre-pruning: no leaf may describe fewer than 50 clients, so leaf rates are not small-sample noise (cf. education=illiterate, 18 rows)."),
            ("class_weight", "balanced", "Only 11.27% subscribe; without reweighting the impurity criterion is dominated by the 'no' class."),
            ("encoding", "one-hot, fitted on train", "sklearn trees need numeric input; one-hot keeps each level as a separate yes/no question."),
            ("random_state", C.RANDOM_STATE, "Ties between equally good splits are broken randomly; fixing the seed makes the tree reproducible."),
        ]),
    })

    # SHAP — TreeExplainer on the fitted tree, aggregated back to original features
    pipe = out["model"]
    prep, tree = pipe.named_steps["prep"], pipe.named_steps["tree"]
    n = min(SAMPLE_CAPS[MID], len(ds.X_test))
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = np.sort(rng.choice(len(ds.X_test), n, replace=False))
    X_raw = ds.X_test.iloc[rows].reset_index(drop=True)
    Xt = prep.transform(X_raw)
    ex = shap.TreeExplainer(tree)
    sv = ex.shap_values(Xt)
    sv = sv[:, :, 1] if sv.ndim == 3 else sv[1]
    names = list(prep.get_feature_names_out())
    feats = list(ds.X_test.columns)
    agg_sv = aggregate_onehot(sv, names, onehot_source_map(names, ds.categorical), feats)
    base = float(np.atleast_1d(ex.expected_value)[-1])
    payload = build_payload(MID, "The decision tree", "TreeExplainer", agg_sv, X_raw, out["proba_test"][rows],
                            ds.y_test[rows], base, "probability (class-balanced)", ds.test_index[rows],
                            extra_note="Because the tree is shallow, most features get exactly zero attribution for most clients — they never appear on that client's path.")
    write_explain(MID, payload)

    m = __import__("json").loads((d / "metrics.json").read_text())
    wo, w = m["runs"]["without_duration"], m["runs"]["with_duration"]
    zero = [f for f, v in agg.items() if v == 0]
    dow_rank = payload["global"]["features"].index("day_of_week") + 1
    hv_ok = all(abs(c["hand"] - c["code"]) < 1e-9 for c in hv["checks"])
    write_json(d / "findings.json", {
        "headline": f"A depth-{out['depth']} tree reaches AUC {wo['roc_auc']:.3f} without the leaky call length — "
                    f"and {w['roc_auc']:.3f} with it.",
        "findings": [
            {"type": "result", "text": f"Without duration: ROC-AUC {wo['roc_auc']:.4f}, PR-AUC {wo['pr_auc']:.4f}, F1 {wo['f1']:.4f} "
                                       f"(threshold {wo['threshold']:.3f} chosen on out-of-fold predictions). The plan's d=5 smoke test measured 0.7902."},
            {"type": "result", "text": f"Leak inflation: +{m['leak_delta']['roc_auc']:.4f} AUC and +{m['leak_delta']['pr_auc']:.4f} PR-AUC when duration is allowed. "
                                       + depth_sentence(out["depth"], out_w["depth"])},
            {"type": "result", "text": f"Hand-computed entropy, information gain, split information and Gini for poutcome "
                                       f"{'match' if hv_ok else 'DO NOT match'} scipy and sklearn to 1e-9. Information gain picks "
                                       f"{hv['c45']['id3_pick']}; C4.5's gain ratio (after the average-gain filter) picks {hv['c45']['pick']}."},
            {"type": "result", "text": f"The depth curve is flat: depth 3 CV AUC {out['curve'][2]['test_roc_auc']:.4f} vs best "
                                       f"{out['cv_roc_auc_mean']:.4f} at depth {out['depth']}. The plan's prediction (shallow ≈ deep) holds — the signal without duration is weak and diffuse."},
            {"type": "result", "text": f"day_of_week ranks {dow_rank} of {len(payload['global']['features'])} in SHAP attribution — the plan predicted near-zero importance; "
                                       f"{'confirmed' if dow_rank > 15 else 'not confirmed for this model: a tree uses it for a few late splits even though its overall rates are nearly flat'}"
                                       + (" (in fact it is never used in a split)." if "day_of_week" in zero else ".")},
            {"type": "limitation", "text": "Axis-parallel splits cannot use the three collinear macro features jointly; the tree picks one (usually nr.employed or euribor3m) and the others become nearly redundant."},
            {"type": "limitation", "text": f"Balanced class weights push accuracy at threshold 0.5 below the 0.8873 majority baseline; at the F1-optimal threshold accuracy is {wo['accuracy']:.4f}. Probabilities are not calibrated (Brier {wo['brier']:.4f})."},
            {"type": "limitation", "text": "A random 80/20 split ignores that the data is time-ordered (May 2008 – Nov 2010); a temporal split would likely score lower."},
        ],
        "comparison_note": "The reference point for every other model. It is the only model whose prediction a human can trace step by step, and the only one whose arithmetic we reproduce by hand.",
        "viva_answer": "Q: Why did you use Gini rather than information gain? — They rarely disagree in practice: we computed both for the root and report them side by side. Gini is cheaper (no logarithms) and tends to isolate the most frequent class; entropy tends to produce more balanced trees. We also show gain ratio, because information gain alone is biased toward high-cardinality features like job, which has 12 levels.",
    })
    validate_model_artifacts(MID)
    print("decision_tree artifacts valid")


if __name__ == "__main__":
    main()
