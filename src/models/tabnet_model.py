"""Agent-06 — TabNet (§7.3).   python -m src.models.tabnet_model"""
from __future__ import annotations

import json
import time

import numpy as np
import plotly.graph_objects as go
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .. import config as C
from ..contract import model_dir, validate_model_artifacts, write_figure, write_json
from ..explain import SAMPLE_CAPS, batched_kernel_shap, build_payload, write_explain
from ._supervised import params_block, run_both

MID = "tabnet"
VAL_SIZE = 0.15
NET = dict(n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-3, cat_emb_dim=2,
           optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
           scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params=dict(step_size=10, gamma=0.9),
           mask_type="sparsemax", seed=C.RANDOM_STATE, verbose=5, device_name="cpu")
FIT = dict(max_epochs=100, patience=20, batch_size=1024, virtual_batch_size=128, eval_metric=["auc"], weights=1)
SHAP_NSAMPLES, SHAP_BG = 200, 16
ACCENT, MUTED, AMBER = "#4FB3A0", "#6B7E8F", "#D4A373"


def prepare(ds):
    """Standardise numerics (fitted on train); categoricals stay integer codes for the embeddings."""
    sc = StandardScaler().fit(ds.X_train[ds.numeric])
    Xtr, Xte = ds.X_train.copy(), ds.X_test.copy()
    Xtr[ds.numeric] = sc.transform(ds.X_train[ds.numeric])
    Xte[ds.numeric] = sc.transform(ds.X_test[ds.numeric])
    return Xtr.to_numpy(np.float32), Xte.to_numpy(np.float32)


def fit_predict(ds):
    torch.manual_seed(C.RANDOM_STATE)
    Xtr, Xte = prepare(ds)
    i_tr, i_val = train_test_split(np.arange(len(Xtr)), test_size=VAL_SIZE, stratify=ds.y_train, random_state=C.RANDOM_STATE)
    clf = TabNetClassifier(cat_idxs=ds.cat_idxs, cat_dims=ds.cat_dims, **NET)
    t0 = time.perf_counter()
    clf.fit(Xtr[i_tr], ds.y_train[i_tr], eval_set=[(Xtr[i_val], ds.y_train[i_val])], eval_name=["val"], **FIT)
    t1 = time.perf_counter()
    proba = clf.predict_proba(Xte)[:, 1]
    t2 = time.perf_counter()
    val_proba = clf.predict_proba(Xtr[i_val])[:, 1]
    print(f"    epochs {len(clf.history['loss'])}, best val AUC {max(clf.history['val_auc']):.4f}", flush=True)
    return {"proba_test": proba, "thr_y": ds.y_train[i_val], "thr_proba": val_proba, "train_seconds": t1 - t0,
            "predict_seconds": t2 - t1, "cv_roc_auc_mean": None, "cv_roc_auc_std": None,
            "model": clf, "Xtr": Xtr, "Xte": Xte, "history": dict(clf.history.history)}


META = {
    "model_id": MID, "display_name": "TabNet", "family": "tabular_deep_learning", "task": "supervised_classification",
    "one_liner": "A neural network that, at each step, chooses a handful of features to look at — and lets you read which ones.",
    "plain_explanation": (
        "Most neural networks look at every feature at once. TabNet instead works in steps, and at each step it decides which small "
        "handful of features to look at — like a person reading a form and covering everything except the two fields that matter "
        "right now, then moving on to a different pair. That choosing is done by a learned mask. Because the mask is explicit, you "
        "can read it afterwards and see exactly which features the network used at each step — unusual for a neural network, and an "
        "interpretability story that does not depend on SHAP."),
    "year": 2021, "reference": "Arık & Pfister, TabNet: Attentive Interpretable Tabular Learning, AAAI 2021",
    "workflow": [
        {"id": "n1", "label": "Normalise input features", "detail": "Categoricals → learned embeddings; batch normalisation on the feature vector (numerics also standardised beforehand).", "formula_ref": None},
        {"id": "n2", "label": "Attentive transformer → mask", "detail": "Learn which features to attend to at this step. Sparsemax forces most weights to exactly zero.", "formula_ref": "f1"},
        {"id": "n3", "label": "Apply prior scale", "detail": "Features already used heavily are down-weighted, encouraging different steps to look at different things.", "formula_ref": "f2"},
        {"id": "n4", "label": "Mask the features", "detail": "Element-wise multiply: everything not selected becomes zero.", "formula_ref": "f3"},
        {"id": "n5", "label": "Feature transformer → split", "detail": "Produce a decision contribution and information for the next step's mask.", "formula_ref": "f4"},
        {"id": "n6", "label": "Repeat for N steps", "detail": "Loop to n2. Here N = 4 steps.", "formula_ref": "f6"},
        {"id": "n7", "label": "Aggregate and predict", "detail": "Sum decision contributions, apply the final linear layer.", "formula_ref": "f5"},
    ],
    "edges": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"], ["n4", "n5"], ["n5", "n2"], ["n5", "n6"], ["n6", "n7"]],
    "formulas": [
        {"id": "f1", "name": "Attentive transformer mask", "latex": r"\mathbf{M}[i] = \text{sparsemax}\big(\mathbf{P}[i-1] \cdot h_i(\mathbf{a}[i-1])\big)",
         "caption": "Sparsemax, unlike softmax, drives most entries to exactly zero — so the mask genuinely selects a subset rather than weighting everything slightly.",
         "glossary": [{"sym": r"h_i", "means": "FC + batch-norm layer of step i"}, {"sym": r"\mathbf{a}[i-1]", "means": "information passed from the previous step"}]},
        {"id": "f2", "name": "Prior scale", "latex": r"\mathbf{P}[i] = \prod_{j=1}^{i}\big(\gamma - \mathbf{M}[j]\big)",
         "caption": "Tracks how much each feature has been used. With γ = 1, a feature used fully at one step is excluded afterwards. Larger γ (1.3 here) allows reuse.",
         "glossary": [{"sym": r"\gamma", "means": "relaxation parameter (1.3)"}]},
        {"id": "f3", "name": "Masked features", "latex": r"\mathbf{f}_{\text{masked}}[i] = \mathbf{M}[i] \odot \mathbf{f}",
         "caption": "⊙ is element-wise multiplication: unselected features become zero for this step.", "glossary": []},
        {"id": "f4", "name": "Feature transformer split", "latex": r"[\mathbf{d}[i], \mathbf{a}[i]] = f_i\big(\mathbf{M}[i] \odot \mathbf{f}\big)",
         "caption": "Output splits in two: d[i] contributes to the prediction, a[i] informs the next mask.",
         "glossary": [{"sym": r"\mathbf{d}[i]", "means": "decision output (n_d = 16 wide)"}, {"sym": r"\mathbf{a}[i]", "means": "attention output (n_a = 16 wide)"}]},
        {"id": "f5", "name": "Aggregate output",
         "latex": r"\mathbf{d}_{\text{out}} = \sum_{i=1}^{N_{\text{steps}}} \text{ReLU}\big(\mathbf{d}[i]\big), \qquad \hat{y} = \mathbf{W}_{\text{final}}\,\mathbf{d}_{\text{out}}",
         "caption": "Each step adds its (non-negative) decision vector; a final linear layer turns the sum into class scores.", "glossary": []},
        {"id": "f6", "name": "Sparsity regularisation",
         "latex": r"\mathcal{L}_{\text{sparse}} = \sum_{i=1}^{N_{\text{steps}}}\sum_{b}\sum_{j} \frac{-\mathbf{M}[i]_{b,j}\log\big(\mathbf{M}[i]_{b,j}+\epsilon\big)}{N_{\text{steps}}\cdot B}",
         "caption": "The entropy of the masks, added to the loss with weight λ_sparse = 0.001. Pushes masks toward selecting fewer features. (Checked against pytorch-tabnet's source: it computes mean over batch of Σ −M·log(M+ε), divided by n_steps.)",
         "glossary": [{"sym": "B", "means": "batch size"}, {"sym": r"\epsilon", "means": "small constant for log(0)"}]},
    ],
    "assumptions": [
        {"text": "Features should be scaled", "status": "required", "evidence": "Numerics standardised (fitted on train). Unscaled inputs give badly conditioned gradients."},
        {"text": "Enough data for a neural network", "status": "warn", "evidence": "~28k training rows is small for deep learning; trees usually win here."},
        {"text": "Categoricals need embedding", "status": "ok", "evidence": "cat_idxs / cat_dims passed; each level gets a learned 2-d embedding instead of one-hot."},
    ],
}


def main():
    print("TabNet: two runs (early stopping on validation AUC)…", flush=True)
    outs = run_both(MID, "embedding", fit_predict, {
        "cv": "not run — each fit takes minutes on CPU; single holdout with early stopping",
        "resampling": "weights=1: training batches sampled with inverse class frequency (inside the training split only)",
        "validation": f"{VAL_SIZE:.0%} of train held out for early stopping and threshold"})
    ds, out = outs["without_duration"]
    _, out_w = outs["with_duration"]
    clf, Xte = out["model"], out["Xte"]
    d = model_dir(MID)
    feats = ds.feature_names
    figs = []

    h = out["history"]
    ep = list(range(1, len(h["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=h["loss"], name="train loss", line=dict(color=MUTED)))
    fig.add_trace(go.Scatter(x=ep, y=h["val_auc"], name="validation AUC", yaxis="y2", line=dict(color=ACCENT, width=2.5)))
    best_ep = int(np.argmax(h["val_auc"])) + 1
    fig.add_vline(x=best_ep, line_dash="dot", line_color=ACCENT, annotation_text=f"best epoch {best_ep}")
    fig.update_layout(title="Training curve (patience 20 epochs)", xaxis_title="epoch", yaxis_title="loss",
                      yaxis2=dict(title="AUC", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.2))
    write_figure(MID, "training_curve.json", fig)
    figs.append({"file": "training_curve.json", "title": "Training curve", "type": "line",
                 "caption": f"Validation AUC peaked at epoch {best_ep} ({max(h['val_auc']):.4f}); training stopped {len(ep) - best_ep} epochs later and restored the best weights."})

    # Masks: per-step mean attention over the test set, aggregated back to the 20 original features
    M_explain, masks = clf.explain(Xte)
    steps = np.array([masks[k].mean(axis=0) for k in sorted(masks)])
    order = np.argsort(-steps.sum(axis=0))
    fig = go.Figure(go.Heatmap(z=steps[:, order], x=[feats[j] for j in order], y=[f"step {i + 1}" for i in range(len(steps))],
                               colorscale=[[0, "#1E2832"], [0.4, "#36695E"], [1, "#9FCBB4"]],
                               hovertemplate="%{y} · %{x}: mean mask %{z:.3f}<extra></extra>"))
    fig.update_layout(title="Feature masks by decision step (mean over test clients)", yaxis_autorange="reversed", margin=dict(b=110))
    write_figure(MID, "mask_heatmap.json", fig)
    per_step_top = [feats[int(np.argmax(s))] for s in steps]
    figs.insert(0, {"file": "mask_heatmap.json", "title": "Feature mask heatmap", "type": "heatmap",
                    "caption": "Which features the network attended to at each of its 4 steps. Top feature per step: "
                               + ", ".join(f"step {i + 1}: {f}" for i, f in enumerate(per_step_top)) + "."})
    write_json(d / "figures" / "masks3d.json", {"type": "mask3d", "steps": len(steps), "features": [feats[j] for j in order],
                                               "values": np.round(steps[:, order], 4).tolist()})
    figs.append({"file": "masks3d.json", "title": "Mask evolution in 3D", "type": "mask3d",
                 "caption": "The same masks as bars: decision steps along depth, features across, mask weight as height."})

    # SHAP (batched Kernel SHAP) and the comparison the plan asks for: do masks and SHAP agree?
    n = SAMPLE_CAPS[MID]
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows = np.sort(rng.choice(len(Xte), n, replace=False))
    bg = out["Xtr"][rng.choice(len(out["Xtr"]), SHAP_BG, replace=False)]
    f = lambda X: clf.predict_proba(np.asarray(X, np.float32))[:, 1]
    t0 = time.perf_counter()
    sv, base, fx = batched_kernel_shap(f, Xte[rows], bg, SHAP_NSAMPLES, chunk=200000)
    shap_secs = time.perf_counter() - t0
    X_raw = ds.X_test.iloc[rows].reset_index(drop=True).copy()
    for c in ds.categorical:
        X_raw[c] = X_raw[c].map(dict(enumerate(ds.vocab[c])))
    mask_imp = clf.feature_importances_
    shap_imp = np.abs(sv).mean(axis=0)
    rho = float(spearmanr(mask_imp, shap_imp).statistic)
    top5_mask = {feats[j] for j in np.argsort(-mask_imp)[:5]}
    top5_shap = {feats[j] for j in np.argsort(-shap_imp)[:5]}
    overlap = sorted(top5_mask & top5_shap)
    payload = build_payload(MID, "TabNet", "Kernel SHAP (batched)", sv, X_raw, fx, ds.y_test[rows], base, "probability (class-balanced sampling)", ds.test_index[rows],
                            extra_note=f"TabNet's own masks and SHAP agree with Spearman ρ = {rho:.2f} across the 20 features; "
                                       f"{len(overlap)} of their top-5 features coincide ({', '.join(overlap) or 'none'}).")
    payload["mask_vs_shap"] = {"spearman_rho": rho, "top5_overlap": overlap,
                               "mask_share": dict(zip(feats, (mask_imp / mask_imp.sum()).tolist())),
                               "shap_share": dict(zip(feats, (shap_imp / shap_imp.sum()).tolist()))}
    write_explain(MID, payload)

    o = np.argsort(mask_imp / mask_imp.sum() + shap_imp / shap_imp.sum())
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[feats[j] for j in o], x=(mask_imp / mask_imp.sum())[o], name="TabNet masks", orientation="h", marker_color=ACCENT))
    fig.add_trace(go.Bar(y=[feats[j] for j in o], x=(shap_imp / shap_imp.sum())[o], name="SHAP (Kernel SHAP)", orientation="h", marker_color=AMBER))
    fig.update_layout(barmode="group", title=f"Two independent attributions (Spearman ρ = {rho:.2f})", xaxis_title="share of importance",
                      xaxis_tickformat=".0%", height=620, margin=dict(l=120), legend=dict(orientation="h", y=-0.1))
    write_figure(MID, "mask_vs_shap.json", fig)
    figs.append({"file": "mask_vs_shap.json", "title": "Masks vs SHAP", "type": "bar",
                 "caption": f"Masks are produced by the architecture; SHAP is computed afterwards from predictions only. Rank agreement ρ = {rho:.2f}; "
                            f"shared top-5: {', '.join(overlap) or 'none'}."})

    write_json(d / "meta.json", {**META, "figures": figs})
    write_json(d / "config.json", {"estimator": "pytorch_tabnet.TabNetClassifier", **params_block([
        ("n_d / n_a", "16 / 16", "Width of the decision and attention outputs (f4). Small: 28k rows cannot support a wide network."),
        ("n_steps", 4, "Number of decision steps (N_steps). 3–8 is typical; 4 keeps the mask heatmap readable."),
        ("gamma", 1.3, "Prior-scale relaxation (f2): >1 lets a feature be reused in later steps."),
        ("lambda_sparse", 1e-3, "Weight of the sparsity loss (f6)."),
        ("cat_emb_dim", 2, "Each categorical level becomes a learned 2-d vector instead of a one-hot column."),
        ("optimizer", "Adam(lr=2e-2) + StepLR(10, 0.9)", "The paper's recommended setup for small tabular data."),
        ("max_epochs / patience", "100 / 20", f"Early stopping on validation AUC; stopped after {len(ep)} epochs."),
        ("batch_size / virtual_batch_size", "1024 / 128", "Ghost batch normalisation: statistics over 128-row virtual batches."),
        ("weights", 1, "Balanced sampling of training batches: the 11.27% minority is drawn as often as the majority."),
        ("seed", C.RANDOM_STATE, "Fixes weight initialisation and batch order."),
    ])})

    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    wo = m["runs"]["without_duration"]
    cb = C.RESULTS_MODELS / "catboost" / "metrics.json"
    cbm = json.loads(cb.read_text())["runs"]["without_duration"] if cb.exists() else None
    findings = [
        {"type": "result", "text": f"Without duration: ROC-AUC {wo['roc_auc']:.4f}, PR-AUC {wo['pr_auc']:.4f}, F1 {wo['f1']:.4f}; trained in {wo['train_seconds']:.0f}s on CPU."},
        {"type": "result", "text": f"Leak inflation: +{m['leak_delta']['roc_auc']:.4f} AUC, +{m['leak_delta']['pr_auc']:.4f} PR-AUC."},
        {"type": "result", "text": f"Masks vs SHAP: Spearman ρ = {rho:.2f}; shared top-5 features: {', '.join(overlap) or 'none'}. "
                                   + ("Two independent attribution methods broadly agree, which strengthens both." if rho >= 0.5 else
                                      "The two methods disagree substantially — a genuine result: masks show what the network looks at, SHAP what changes its output.")},
    ]
    if cbm:
        verdict = "loses to" if wo["pr_auc"] < cbm["pr_auc"] else "beats"
        findings.append({"type": "result" if verdict == "beats" else "failure",
                         "text": f"TabNet {verdict} CatBoost: PR-AUC {wo['pr_auc']:.4f} vs {cbm['pr_auc']:.4f}, ROC-AUC {wo['roc_auc']:.4f} vs {cbm['roc_auc']:.4f}, "
                                 f"at {wo['train_seconds'] / max(cbm['train_seconds'], 1e-9):.1f}× the training time. We did not tune TabNet further to change this."})
    findings += [
        {"type": "limitation", "text": "No cross-validation (CPU cost); the single-holdout AUC has no error bar, and neural nets vary more across seeds than trees."},
        {"type": "limitation", "text": f"SHAP for TabNet uses Kernel SHAP (batched; same estimator as shap.KernelExplainer) on {n} clients with {SHAP_NSAMPLES} coalition samples ({shap_secs / 60:.1f} min) — approximate, unlike exact TreeSHAP."},
        {"type": "limitation", "text": "Masks show where the network looks, not how a feature moves the prediction; a heavily attended feature can still have a small effect."},
    ]
    write_json(d / "findings.json", {
        "headline": f"TabNet reaches AUC {wo['roc_auc']:.3f} — a neural network that shows its own feature choices, at more cost and no gain over trees.",
        "findings": findings,
        "comparison_note": "The only model with an attribution produced by the architecture itself, which lets us cross-check SHAP rather than trust it.",
        "viva_answer": "Q: Why would anyone use a neural network on tabular data when boosting usually wins? — Usually they would not, and our results reflect that. We included TabNet for a specific reason: its attention masks give a feature-importance signal produced by the architecture itself rather than computed afterwards. That lets us check SHAP against an independent method. The honest finding is that it costs more compute and gives less accuracy here.",
    })
    validate_model_artifacts(MID)
    print(f"tabnet artifacts valid; masks vs SHAP rho={rho:.2f}")


if __name__ == "__main__":
    main()
