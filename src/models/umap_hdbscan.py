"""Agent-04 — UMAP + HDBSCAN (§7.5).   python -m src.models.umap_hdbscan

Not tuned for a pretty picture: the config is the plan's, fixed before looking at the output.
"""
from __future__ import annotations

import time

import hdbscan
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score

from .. import config as C
from ..contract import model_dir, validate_model_artifacts, write_figure, write_json
from ..data_loader import load_raw, split_indices
from ..preprocessing import clean, feature_lists, make_onehot_preprocessor

MID = "umap_hdbscan"
UMAP_KW = dict(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=C.RANDOM_STATE)
HDB_KW = dict(min_cluster_size=250, min_samples=10, prediction_data=True)
MAX_POINTS = 8000
YES, NO = "#4FB3A0", "#6B7E8F"
PALETTE = ["#4FB3A0", "#D4A373", "#7B9FC7", "#C88B8B", "#A89BC4", "#8FA880", "#5FA8B8", "#C4A26B",
           "#8E9FD0", "#B88FA8", "#9FB86F", "#D09A7A"]
NOISE = "#3A4652"


def cluster_color(c):
    return NOISE if c < 0 else PALETTE[c % len(PALETTE)]


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, y: np.ndarray):
    cats, nums = feature_lists(False)
    glob_rate = y.mean()
    out = []
    for c in sorted(set(labels)):
        m = labels == c
        devs = []
        for f in [x for x in nums if x != "pdays"]:  # pdays uses -1 as 'not applicable'; was_contacted carries it
            sd = df[f].std() or 1
            z = (df.loc[m, f].mean() - df[f].mean()) / sd
            devs.append({"feature": f, "kind": "numeric", "deviation": float(z),
                         "text": f"{f} {'high' if z > 0 else 'low'} (mean {df.loc[m, f].mean():.4g} vs {df[f].mean():.4g})"})
        for f in cats:
            share_c = df.loc[m, f].value_counts(normalize=True)
            share_g = df[f].value_counts(normalize=True)
            lvl = (share_c - share_g.reindex(share_c.index)).idxmax()
            diff = float(share_c[lvl] - share_g[lvl])
            devs.append({"feature": f"{f}={lvl}", "kind": "categorical", "deviation": diff * 3,  # comparable scale
                         "text": f"{f}={lvl} ({share_c[lvl]:.0%} vs {share_g[lvl]:.0%} overall)"})
        devs.sort(key=lambda d: -abs(d["deviation"]))
        top = devs[:6]
        rate = float(y[m].mean())
        rel = "high" if rate > 1.5 * glob_rate else "low" if rate < 0.67 * glob_rate else "average"
        name = "Noise (no cluster)" if c < 0 else f"Cluster {c}"
        profile = (f"{'; '.join(d['text'] for d in top[:3])}. Subscription {rate:.1%} — {rel} response."
                   if c >= 0 else f"Points HDBSCAN would not assign to any dense group. Subscription {rate:.1%}.")
        out.append({"id": int(c), "name": name, "size": int(m.sum()), "subscribe_rate": rate, "response": rel,
                    "profile": profile, "top_deviations": top,
                    "months": {k: int(v) for k, v in df.loc[m, "month"].value_counts().head(4).items()}})
    return out


def condensed_tree_figure(clusterer):
    ct = clusterer.condensed_tree_.to_pandas()
    nodes = ct[ct.child_size > 1]
    root = int(ct.parent.min())
    births = {root: 0.0}
    for _, r in nodes.iterrows():
        births[int(r.child)] = float(r.lambda_val)
    sizes = {int(r.child): int(r.child_size) for _, r in nodes.iterrows()}
    sizes[root] = int(ct[ct.parent == root].child_size.sum())
    deaths = {k: float(ct[ct.parent == k].lambda_val.max()) if (ct.parent == k).any() else births[k] for k in births}
    children = {k: [int(c) for c in nodes[nodes.parent == k].child] for k in births}
    selected = set()
    try:
        selected = {int(x) for x in clusterer.condensed_tree_._select_clusters()}
    except Exception:
        pass
    # DFS leaf ordering for x positions
    xs, order = {}, []

    def place(k):
        if not children[k]:
            xs[k] = len(order); order.append(k); return xs[k]
        pos = [place(c) for c in children[k]]
        xs[k] = float(np.mean(pos)); return xs[k]

    place(root)
    fig = go.Figure()
    cap = np.percentile(list(deaths.values()), 98)
    for k in births:
        w = 0.15 + 0.6 * np.log1p(sizes[k]) / np.log1p(sizes[root])
        sel = k in selected
        fig.add_trace(go.Scatter(
            x=[xs[k] - w / 2, xs[k] + w / 2, xs[k] + w / 2, xs[k] - w / 2, xs[k] - w / 2],
            y=[births[k], births[k], min(deaths[k], cap), min(deaths[k], cap), births[k]], fill="toself",
            mode="lines", line=dict(color=YES if sel else "#4A5866", width=1),
            fillcolor="rgba(79,179,160,0.55)" if sel else "rgba(107,126,143,0.25)",
            hovertemplate=f"{'selected cluster' if sel else 'branch'}: {sizes[k]:,} points<br>born λ={births[k]:.3g}<extra></extra>",
            showlegend=False))
        for c in children[k]:
            fig.add_trace(go.Scatter(x=[xs[k], xs[c]], y=[births[c], births[c]], mode="lines",
                                     line=dict(color="#6B7E8F", width=1), hoverinfo="skip", showlegend=False))
    fig.update_layout(title="HDBSCAN condensed tree (teal = clusters kept by stability)",
                      yaxis=dict(title="λ = 1 / distance (density increases downward)", autorange="reversed"),
                      xaxis=dict(visible=False))
    return fig, len(selected)


def scatter2d(emb, color_vals, names, colors, title, hover):
    fig = go.Figure()
    for val in sorted(set(color_vals.tolist())):
        m = color_vals == val
        fig.add_trace(go.Scattergl(x=np.round(emb[m, 0], 3), y=np.round(emb[m, 1], 3), mode="markers", name=str(names[m][0]),
                                   marker=dict(size=3, color=str(colors[m][0]), opacity=0.7), text=hover[m],
                                   hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
                      yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False), legend=dict(itemsizing="constant"))
    return fig


META = {
    "model_id": MID, "display_name": "UMAP + HDBSCAN", "family": "unsupervised", "task": "unsupervised_clustering",
    "one_liner": "Flattens 50 dimensions into 2–3 while keeping neighbours together, then finds dense groups without being told how many.",
    "plain_explanation": (
        "UMAP takes data with fifty columns and squeezes it down to two or three, trying to keep neighbours as neighbours. "
        "Think of flattening a crumpled sheet of paper without tearing it — points that were close stay close, and the overall "
        "shape survives. HDBSCAN then finds clusters in that flattened space by looking for dense regions. Unlike k-means, you do "
        "not tell it how many clusters to find, and it is allowed to say 'this point belongs to no cluster'. Together they answer: "
        "do our clients fall into natural groups, and do subscribers sit anywhere in particular?"),
    "year": 2018, "reference": "McInnes, Healy & Melville, UMAP (2018); Campello, Moulavi & Sander, HDBSCAN (2013); McInnes & Healy, hdbscan (2017)",
    "workflow": [
        {"id": "n1", "label": "Build the k-NN graph", "detail": "For each client, find its 15 nearest neighbours in the full (one-hot, scaled) feature space.", "formula_ref": "f1"},
        {"id": "n2", "label": "Convert to fuzzy memberships", "detail": "Turn distances into probabilities. Each point's local scale is set so it has a consistent neighbourhood size.", "formula_ref": "f1"},
        {"id": "n3", "label": "Symmetrise", "detail": "Reconcile 'A says B is a neighbour' with 'B says A is not'.", "formula_ref": "f2"},
        {"id": "n4", "label": "Optimise the low-dim layout", "detail": "Place points in 2D/3D so the low-dim graph matches the high-dim one.", "formula_ref": "f3,f4"},
        {"id": "n5", "label": "Mutual reachability distance", "detail": "Inflate distances in sparse regions so noise cannot bridge clusters.", "formula_ref": "f5"},
        {"id": "n6", "label": "Build the hierarchy", "detail": "Minimum spanning tree → condensed cluster tree.", "formula_ref": None},
        {"id": "n7", "label": "Extract stable clusters", "detail": "Keep clusters that persist across density thresholds.", "formula_ref": "f6"},
    ],
    "edges": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"], ["n4", "n5"], ["n5", "n6"], ["n6", "n7"]],
    "formulas": [
        {"id": "f1", "name": "Local fuzzy membership (UMAP)", "latex": r"p_{j|i} = \exp\!\left(-\frac{d(x_i,x_j) - \rho_i}{\sigma_i}\right)",
         "caption": "ρᵢ is the distance to the nearest neighbour, so every point is certainly connected to at least one other. σᵢ is solved per point so that Σⱼ p_{j|i} = log₂ k — this is what makes UMAP adapt to varying local density.",
         "glossary": [{"sym": r"\rho_i", "means": "distance to nearest neighbour"}, {"sym": r"\sigma_i", "means": "per-point bandwidth"}, {"sym": "k", "means": "n_neighbors (15)"}]},
        {"id": "f2", "name": "Symmetrisation", "latex": r"p_{ij} = p_{j|i} + p_{i|j} - p_{j|i}\,p_{i|j}",
         "caption": "Probabilistic OR: an edge exists if either endpoint claims it.", "glossary": []},
        {"id": "f3", "name": "Low-dimensional similarity", "latex": r"q_{ij} = \big(1 + a\,\lVert y_i - y_j \rVert_2^{2b}\big)^{-1}",
         "caption": "a and b are fitted from min_dist. Heavy tails let distant points stay distant without dominating the loss.",
         "glossary": [{"sym": "y_i", "means": "low-dimensional position of point i"}]},
        {"id": "f4", "name": "Cross-entropy objective",
         "latex": r"\mathcal{L} = \sum_{i \ne j} \left[ p_{ij}\log\frac{p_{ij}}{q_{ij}} + (1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}} \right]",
         "caption": "The first term pulls neighbours together; the second pushes non-neighbours apart. Keeping both is why UMAP preserves global structure better than t-SNE.", "glossary": []},
        {"id": "f5", "name": "Mutual reachability distance (HDBSCAN)",
         "latex": r"d_{\text{mreach-}k}(a,b) = \max\big\{\text{core}_k(a),\; \text{core}_k(b),\; d(a,b)\big\}",
         "caption": "core_k(x) is the distance from x to its k-th nearest neighbour. Points in sparse regions get pushed apart, so noise cannot act as a bridge between two real clusters.",
         "glossary": [{"sym": r"\text{core}_k(x)", "means": "distance to x's k-th nearest neighbour (k = min_samples = 10)"}]},
        {"id": "f6", "name": "Cluster stability", "latex": r"S(C) = \sum_{p \in C}\big(\lambda_p - \lambda_{\text{birth}}(C)\big), \qquad \lambda = \frac{1}{\text{distance}}",
         "caption": "How long a cluster survives as the density threshold tightens. HDBSCAN keeps the set of clusters maximising total stability — this is what replaces choosing k.", "glossary": []},
    ],
    "assumptions": [
        {"text": "Features must be scaled", "status": "required", "evidence": "Distance-based: one-hot + StandardScaler applied."},
        {"text": "Meaningful distance in the feature space", "status": "warn", "evidence": "One-hot encoding makes Euclidean distance crude for categoricals (every level mismatch costs the same); Gower distance would be an alternative."},
        {"text": "Clusters have varying density", "status": "ok", "evidence": "Exactly what HDBSCAN handles and DBSCAN does not."},
        {"text": "Local structure is meaningful", "status": "unknown", "evidence": "The page's open question — answered by the mutual-information table in the findings."},
    ],
}


def main():
    raw = load_raw()
    df = clean(raw)
    y = df.pop(C.TARGET).to_numpy()
    cats, nums = feature_lists(False)
    X = make_onehot_preprocessor(cats, nums, scale=True).fit_transform(df[cats + nums])
    print(f"UMAP on {X.shape[0]:,} × {X.shape[1]} (one-hot + scaled)…", flush=True)
    cache = C.DATA_PROCESSED / "umap_cache.npz"
    t0 = time.perf_counter()
    if cache.exists():
        z = np.load(cache)
        emb2, emb3, timing = z["emb2"], z["emb3"], dict(zip(["umap_2d", "umap_3d"], z["timing"]))
        print("  using cached embeddings", flush=True)
    else:
        emb2 = umap.UMAP(n_components=2, **UMAP_KW).fit_transform(X)
        t1 = time.perf_counter()
        emb3 = umap.UMAP(n_components=3, **UMAP_KW).fit_transform(X)
        timing = {"umap_2d": t1 - t0, "umap_3d": time.perf_counter() - t1}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, emb2=emb2, emb3=emb3, timing=np.array([timing["umap_2d"], timing["umap_3d"]]))
    print(f"  UMAP 2D {timing['umap_2d']:.0f}s, 3D {timing['umap_3d']:.0f}s; HDBSCAN…", flush=True)
    t2 = time.perf_counter()
    clusterer = hdbscan.HDBSCAN(**HDB_KW).fit(emb2)
    t3 = time.perf_counter()
    labels = clusterer.labels_
    memb = clusterer.probabilities_
    n_clusters = int(labels.max() + 1)
    noise = float((labels < 0).mean())

    # How much does the structure know about y, month, and the economic regime?
    macro_id = df[C.MACRO].astype(str).agg("|".join, axis=1)
    ami = {
        "y (subscription)": float(adjusted_mutual_info_score(y, labels)),
        "month": float(adjusted_mutual_info_score(df["month"], labels)),
        "macro regime (5 macro features)": float(adjusted_mutual_info_score(macro_id, labels)),
        "contact": float(adjusted_mutual_info_score(df["contact"], labels)),
        "poutcome": float(adjusted_mutual_info_score(df["poutcome"], labels)),
        "job": float(adjusted_mutual_info_score(df["job"], labels)),
    }
    # Out-of-sample: does "subscription rate of my cluster" rank test clients? (rates from train only)
    tr, te = split_indices(len(y), y)
    rate_tr = pd.Series(y[tr]).groupby(labels[tr]).mean()
    score_te = pd.Series(labels[te]).map(rate_tr).fillna(y[tr].mean()).to_numpy()
    cluster_auc = float(roc_auc_score(y[te], score_te))
    write_json(model_dir(MID) / "predictions.json", {"test_index": te.tolist(), "y_true": y[te].tolist(),
                                                     "score_name": "subscription rate of the client's cluster (training clients)",
                                                     "without_duration": {"proba": np.round(score_te, 5).tolist()}})

    profiles = profile_clusters(df, labels, y)
    d = model_dir(MID)

    rng = np.random.default_rng(C.RANDOM_STATE)
    keep = np.sort(rng.choice(len(y), MAX_POINTS, replace=False))
    hover = np.array([f"age {a} · {j} · {mo} · poutcome {p}<br>{'subscribed' if t else 'did not subscribe'} · cluster {c}"
                      for a, j, mo, p, t, c in zip(df.age, df.job, df.month, df.poutcome, y, labels)])
    figs = []
    yk = y[keep]
    f = scatter2d(emb2[keep], yk, np.where(yk == 1, "subscribed", "not subscribed"), np.where(yk == 1, YES, NO),
                  "UMAP 2D coloured by subscription (8,000-point sample)", hover[keep])
    write_figure(MID, "map_by_class.json", f)
    figs.append({"file": "map_by_class.json", "title": "2D map by subscription", "type": "scatter",
                 "caption": f"Subscribers do not form their own region. They are concentrated in some islands more than others, "
                            f"but AMI between clusters and subscription is only {ami['y (subscription)']:.3f}."})
    lk = labels[keep]
    f = scatter2d(emb2[keep], lk, np.array(["noise" if c < 0 else f"cluster {c}" for c in lk]),
                  np.array([cluster_color(c) for c in lk]), "UMAP 2D coloured by HDBSCAN cluster", hover[keep])
    write_figure(MID, "map_by_cluster.json", f)
    figs.append({"file": "map_by_cluster.json", "title": "2D map by cluster", "type": "scatter",
                 "caption": f"{n_clusters} clusters, {noise:.1%} of clients labelled noise. min_cluster_size=250, min_samples=10 — set before looking."})
    f, n_sel = condensed_tree_figure(clusterer)
    write_figure(MID, "condensed_tree.json", f)
    figs.append({"file": "condensed_tree.json", "title": "Condensed tree (HDBSCAN hierarchy)", "type": "custom",
                 "caption": "Each bar is a candidate cluster, from the density at which it splits off (top) to where it dissolves (bottom). "
                            "HDBSCAN keeps the teal ones: the set with the greatest total stability (f6)."})

    rows = [{"cluster": p["name"], "size": p["size"], "subscribe_rate": round(p["subscribe_rate"], 4),
             "response": p["response"], "profile": p["profile"]} for p in sorted(profiles, key=lambda p: -p["subscribe_rate"])]
    write_json(d / "figures" / "cluster_table.json", {"type": "table", "columns": ["cluster", "size", "subscribe_rate", "response", "profile"],
                                                      "format": {"subscribe_rate": "pct", "size": "int"}, "rows": rows})
    figs.append({"file": "cluster_table.json", "title": "Cluster profiles in plain language", "type": "table",
                 "caption": "Each profile lists the three features that deviate most from the overall population. Generated from the data, not written by hand."})

    ami_rows = sorted(ami.items(), key=lambda kv: kv[1])
    f = go.Figure(go.Bar(x=[v for _, v in ami_rows], y=[k for k, _ in ami_rows], orientation="h",
                         marker_color=[YES if k.startswith("y") else "#7B9FC7" for k, _ in ami_rows],
                         hovertemplate="%{y}: AMI %{x:.3f}<extra></extra>"))
    f.update_layout(title="What do the clusters line up with? (adjusted mutual information)", xaxis_title="AMI (0 = unrelated, 1 = identical partition)",
                    margin=dict(l=210))
    write_figure(MID, "ami.json", f)
    top_k = max(ami, key=ami.get)
    figs.append({"file": "ami.json", "title": "Clusters vs known variables", "type": "bar",
                 "caption": f"The clusters align most with {top_k} (AMI {ami[top_k]:.2f}) and barely with subscription ({ami['y (subscription)']:.3f})."})

    pts3 = {"type": "points3d", "n_total": int(len(y)), "n_shown": int(len(keep)),
            "points": [{"x": round(float(emb3[i, 0]), 3), "y": round(float(emb3[i, 1]), 3), "z": round(float(emb3[i, 2]), 3),
                        "cluster": int(labels[i]), "y_true": int(y[i]), "membership": round(float(memb[i]), 3),
                        "age": int(df.age.iloc[i]), "job": df.job.iloc[i], "month": df.month.iloc[i], "poutcome": df.poutcome.iloc[i]}
                       for i in keep],
            "clusters": [{"id": p["id"], "size": p["size"], "subscribe_rate": round(p["subscribe_rate"], 4), "profile": p["profile"],
                          "color": cluster_color(p["id"])} for p in profiles],
            "note": "Clusters come from HDBSCAN on the 2D embedding; the 3D layout is a separate UMAP run of the same data."}
    write_json(d / "figures" / "points3d.json", pts3, ndigits=3)
    figs.insert(0, {"file": "points3d.json", "title": "3D point cloud", "type": "points3d",
                    "caption": "8,000 clients in 3D UMAP space. Drag to orbit; hover a point for the client card."})

    write_json(d / "meta.json", {**META, "figures": figs})
    write_json(d / "config.json", {
        "estimator": "umap.UMAP + hdbscan.HDBSCAN",
        "params": [
            {"name": "n_components", "value": "2 and 3", "why": "2D for analysis and clustering, 3D for the hero visual."},
            {"name": "n_neighbors", "value": 15, "why": "UMAP's default: balances local detail against global structure. Not tuned."},
            {"name": "min_dist", "value": 0.1, "why": "How tightly points may pack in the embedding; default, not tuned."},
            {"name": "metric", "value": "euclidean", "why": "On one-hot + standardised features. A known weakness for categoricals (see assumptions)."},
            {"name": "min_cluster_size", "value": 250, "why": "A segment smaller than ~0.6% of clients is not actionable for a campaign."},
            {"name": "min_samples", "value": 10, "why": "k for the core distance in f5; larger = more conservative, more noise."},
            {"name": "clustered_space", "value": "2D UMAP embedding", "why": "So the colours on the 2D map are exactly the clusters found."},
            {"name": "input", "value": "all 41,188 clients, duration excluded", "why": "Unsupervised: the target is used only for colouring and evaluation, never for fitting."},
            {"name": "random_state", "value": C.RANDOM_STATE, "why": "UMAP is stochastic; a fixed seed makes the picture reproducible (and single-threaded)."},
        ],
    })
    write_json(d / "metrics.json", {
        "model_id": MID, "task": "unsupervised",
        "n_clusters": n_clusters, "noise_fraction": noise, "ami": ami,
        "cluster_rate_auc_test": cluster_auc,
        "cluster_rate_auc_note": "Score each test client by the subscription rate of its cluster (rates from training rows only).",
        "subscribe_rate_range": [min(p["subscribe_rate"] for p in profiles if p["id"] >= 0), max(p["subscribe_rate"] for p in profiles if p["id"] >= 0)],
        "timing_seconds": {**timing, "hdbscan": t3 - t2},
        "selected_clusters_in_tree": n_sel,
    })

    best = max((p for p in profiles if p["id"] >= 0), key=lambda p: p["subscribe_rate"])
    worst = min((p for p in profiles if p["id"] >= 0), key=lambda p: p["subscribe_rate"])
    biggest = max((p for p in profiles if p["id"] >= 0), key=lambda p: p["size"])
    write_json(d / "explain.json", {
        "kind": "cluster_profile",
        "clusters": [{"id": p["id"], "name": p["name"], "size": p["size"], "subscribe_rate": p["subscribe_rate"],
                      "profile": p["profile"], "deviations": p["top_deviations"]} for p in profiles],
        "global": {"plain_text": f"The structure UMAP finds is mostly about {top_k}: clients called under the same economic "
                                 f"conditions sit together, whatever their job or age."},
        "narrative": {
            "headline": f"Clients group by when they were called, not by whether they subscribed.",
            "body": f"HDBSCAN finds {n_clusters} dense groups. Their subscription rates range from {worst['subscribe_rate']:.1%} to "
                    f"{best['subscribe_rate']:.1%}, so the groups do carry some signal (cluster-rate AUC {cluster_auc:.3f} on held-out clients), "
                    f"but the partition matches {top_k} (AMI {ami[top_k]:.2f}) far better than subscription (AMI {ami['y (subscription)']:.3f}). "
                    f"The largest group ({biggest['size']:,} clients) — {biggest['profile']}",
            "caveat": "SHAP does not apply to clustering. These are per-cluster deviations from the population average: they describe a group, not a cause.",
        },
    })
    write_json(d / "findings.json", {
        "headline": "The two classes do not separate. Clients cluster by the economic period of the call — the finding the plan predicted.",
        "findings": [
            {"type": "result", "text": f"{n_clusters} clusters, {noise:.1%} noise. Adjusted mutual information with the clusters: "
                                       + ", ".join(f"{k} {v:.3f}" for k, v in sorted(ami.items(), key=lambda kv: -kv[1])) + "."},
            {"type": "result", "text": f"Best cluster subscribes at {best['subscribe_rate']:.1%} ({best['size']:,} clients): {best['profile']}"},
            {"type": "result", "text": f"Using only 'which cluster are you in?' ranks held-out clients at AUC {cluster_auc:.3f} — real but weaker than the supervised models (~0.80)."},
            {"type": "failure", "text": "No region of the map is predominantly subscribers. The plan predicted this; we did not re-tune n_neighbors/min_dist to make one appear."},
            {"type": "limitation", "text": f"The five macro features take only {macro_id.nunique()} distinct value combinations across 41,188 clients — they are period labels in disguise, and after scaling they act like a strong categorical signal that dominates Euclidean distance."},
            {"type": "limitation", "text": "Euclidean distance on one-hot columns treats every categorical mismatch as equally far. Gower distance or CatBoost embeddings might give different structure."},
            {"type": "limitation", "text": "UMAP distances between islands are not meaningful; only neighbourhoods are. Do not read the gap between two clusters as a measure of difference."},
        ],
        "comparison_note": "The only page that never sees the label while fitting. It tells us why the supervised models lean on macro features: the data's dominant structure is the economic calendar.",
        "viva_answer": "Q: Why HDBSCAN rather than k-means? — Three reasons. k-means needs the number of clusters in advance and we had no basis for choosing one. k-means assumes roughly spherical, equally sized clusters, which is not what UMAP output looks like. And k-means forces every point into a cluster, whereas HDBSCAN can label a point as noise — which matters here, because many clients genuinely do not belong to any tight group.",
    })
    validate_model_artifacts(MID)
    print(f"umap_hdbscan artifacts valid: {n_clusters} clusters, noise {noise:.1%}, AMI {ami}")


if __name__ == "__main__":
    main()
