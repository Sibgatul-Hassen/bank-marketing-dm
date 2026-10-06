"""Agent-07 — FP-Growth association rules (§7.6).   python -m src.models.fpgrowth_model"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.metrics import roc_auc_score

from .. import config as C
from ..contract import model_dir, validate_model_artifacts, write_figure, write_json
from ..data_loader import load_raw, split_indices
from ..preprocessing import clean

MID = "fp_growth"
MIN_SUPPORT, MIN_CONF, MIN_LIFT, MAX_LEN = 0.01, 0.30, 1.2, 4
MIN_IMPROVEMENT = 0.02  # Bayardo et al. (1999): a rule must beat every simpler sub-rule's confidence by 2 points
RUNTIME_SUPPORTS = [0.10, 0.05, 0.03, 0.02, 0.01]
ACCENT, MUTED, AMBER, BLUE, ROSE = "#4FB3A0", "#6B7E8F", "#D4A373", "#7B9FC7", "#C88B8B"


# ------------------------------------------------------------------ baskets

def make_baskets(df: pd.DataFrame) -> pd.DataFrame:
    items = pd.DataFrame(index=df.index)
    decade = (df["age"] // 10 * 10).clip(upper=70)
    items["age"] = decade.map(lambda d: "70+" if d >= 70 else f"{int(d)}s")
    for c in ["job", "marital", "education", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]:
        items[c] = df[c]
    items["default"] = np.where(df["default_unknown"] == 1, "unknown", "known")
    items["campaign"] = pd.cut(df["campaign"], [0, 1, 3, np.inf], labels=["1", "2-3", "4+"]).astype(str)
    for c in C.MACRO:
        # Terciles where ties allow: macro features take few distinct values, so edges can coincide.
        edges = np.unique(np.quantile(df[c], [0, 1 / 3, 2 / 3, 1]))
        names = {2: ["low", "high"], 3: ["low", "mid", "high"]}.get(len(edges) - 1, ["all"])
        items[c] = pd.cut(df[c], edges, labels=names, include_lowest=True).astype(str)
    items["y"] = np.where(df[C.TARGET] == 1, "yes", "no")
    onehot = pd.get_dummies(items, prefix_sep="=").astype(bool)
    return onehot


# ------------------------------------------------------------------ our own level-wise Apriori (for L1–L4 counts)

def own_apriori(B: pd.DataFrame, min_support: float, max_len: int):
    n = len(B)
    min_count = int(np.ceil(min_support * n))
    packed = np.packbits(B.to_numpy().T, axis=1)                      # items × bytes
    pad = (-packed.shape[1]) % 8
    words = np.pad(packed, ((0, 0), (0, pad))).view(np.uint64)         # items × uint64 words

    def support(cands):
        acc = words[cands[:, 0]].copy()
        for j in range(1, cands.shape[1]):
            acc &= words[cands[:, j]]
        return np.bitwise_count(acc).sum(axis=1)

    levels = []
    c1 = np.arange(B.shape[1])[:, None]
    s1 = support(c1)
    L = [tuple(c) for c, s in zip(c1, s1) if s >= min_count]
    levels.append({"k": 1, "candidates": int(B.shape[1]), "pruned_by_subset": 0,
                   "pruned_by_support": int(B.shape[1] - len(L)), "frequent": len(L)})
    k = 2
    while L and k <= max_len:
        Lset = set(L)
        by_prefix = {}
        for it in L:
            by_prefix.setdefault(it[:-1], []).append(it[-1])
        joined = []
        for prefix, tails in by_prefix.items():
            tails.sort()
            for a, b in itertools.combinations(tails, 2):
                joined.append(prefix + (a, b))
        survivors = [c for c in joined if all(sub in Lset for sub in itertools.combinations(c, k - 1))]
        if survivors:
            arr = np.array(survivors)
            sup = np.concatenate([support(arr[i:i + 20000]) for i in range(0, len(arr), 20000)])
            newL = [tuple(c) for c, s in zip(arr, sup) if s >= min_count]
        else:
            newL = []
        levels.append({"k": k, "candidates": len(joined), "pruned_by_subset": len(joined) - len(survivors),
                       "pruned_by_support": len(survivors) - len(newL), "frequent": len(newL)})
        L = newL
        k += 1
    return levels


# ------------------------------------------------------------------ helpers

def fs(s):
    return " + ".join(sorted(s))


def rule_record(r, n):
    return {"antecedent": sorted(r["antecedents"]), "consequent": sorted(r["consequents"]),
            "support": float(r["support"]), "confidence": float(r["confidence"]), "lift": float(r["lift"]),
            "coverage": float(r["antecedent support"]), "leverage": float(r["leverage"]),
            "conviction": None if not np.isfinite(r["conviction"]) else float(r["conviction"]),
            "count": int(round(r["support"] * n))}


def speed_sentence(runtime):
    last = runtime[-1]
    ratio = last["apriori"] / max(last["fpgrowth"], 1e-9)
    per = ", ".join(f"{r['min_support']:.0%}: {r['apriori'] / max(r['fpgrowth'], 1e-9):.2f}×" for r in runtime)
    if ratio >= 1.2:
        head = f"FP-Growth is {ratio:.1f}× faster than mlxtend's (low-memory) Apriori at 1% support for identical output."
    elif ratio <= 0.83:
        head = f"Contrary to the textbook expectation, mlxtend's low-memory Apriori was {1 / ratio:.1f}× faster than its FP-Growth at 1% support."
    else:
        head = f"FP-Growth and mlxtend's low-memory Apriori took about the same time at 1% support ({last['fpgrowth']:.0f}s vs {last['apriori']:.0f}s)."
    return head + f" Apriori/FP-Growth time ratio by support — {per}. With only 77 items and itemsets capped at 4, candidate generation stays cheap; FP-Growth's advantage grows with longer patterns."


def test_stats(B_test, ante, cons):
    a = B_test[list(ante)].all(axis=1)
    ac = a & B_test[list(cons)].all(axis=1)
    cs = B_test[list(cons)].all(axis=1).mean()
    conf = ac.sum() / a.sum() if a.sum() else float("nan")
    return float(conf), float(conf / cs) if a.sum() else float("nan"), int(a.sum())


META = {
    "model_id": MID, "display_name": "FP-Growth", "family": "pattern_mining", "task": "association_rules",
    "one_liner": "Finds attribute combinations that occur often, then keeps the rules that beat chance — ranked by lift, not confidence.",
    "plain_explanation": (
        "Association rule mining looks for statements like 'clients who were contacted on a mobile AND whose previous campaign "
        "succeeded tend to subscribe.' It does this in two stages: first find the combinations of attributes that occur often "
        "enough to matter, then turn those into directional rules and keep the trustworthy ones. Apriori does this by generating "
        "candidate combinations level by level and scanning the data at each level. FP-Growth builds a compressed tree of the data "
        "in two passes and mines it recursively, generating no candidates at all — usually much faster for the same result."),
    "year": 2000, "reference": "Han, Pei & Yin, Mining frequent patterns without candidate generation, SIGMOD 2000; Agrawal & Srikant, Apriori, VLDB 1994",
    "workflow": [
        {"id": "n1", "label": "Bin continuous features", "detail": "age into decades, campaign into {1, 2–3, 4+}, macro features into terciles.", "formula_ref": None},
        {"id": "n2", "label": "Build transaction baskets", "detail": "Each client becomes a set of items: job=student, month=mar, y=yes.", "formula_ref": None},
        {"id": "n3", "label": "First scan — item counts", "detail": "Count every single item, discard those below minimum support.", "formula_ref": "f1"},
        {"id": "n4", "label": "Second scan — build FP-tree", "detail": "Insert each transaction with items sorted by frequency; shared prefixes merge into one path.", "formula_ref": "f6"},
        {"id": "n5", "label": "Mine conditional pattern bases", "detail": "For each item, extract its conditional tree and recurse. No candidate generation.", "formula_ref": None},
        {"id": "n6", "label": "Generate rules", "detail": "For each frequent itemset, split into antecedent and consequent; keep those above minimum confidence.", "formula_ref": "f2"},
        {"id": "n7", "label": "Rank by lift", "detail": "Sort by lift, not confidence — and report coverage beside it.", "formula_ref": "f3,f4,f5"},
    ],
    "edges": [["n1", "n2"], ["n2", "n3"], ["n3", "n4"], ["n4", "n5"], ["n5", "n6"], ["n6", "n7"]],
    "formulas": [
        {"id": "f1", "name": "Support", "latex": r"\text{supp}(X) = \frac{|\{T \in D : X \subseteq T\}|}{|D|}",
         "caption": "The fraction of clients whose record contains all of X. Below minimum support (1% here), a pattern is too rare to act on.",
         "glossary": [{"sym": "T", "means": "one client's basket of items"}, {"sym": "D", "means": "all baskets"}]},
        {"id": "f2", "name": "Confidence", "latex": r"\text{conf}(X \Rightarrow Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)} = P(Y \mid X)",
         "caption": "How often the rule holds when X is present.", "glossary": []},
        {"id": "f3", "name": "Lift", "latex": r"\text{lift}(X \Rightarrow Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)\,\text{supp}(Y)} = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)}",
         "caption": "How much more likely Y is given X than by chance. Lift > 1: positive association · = 1: independent — the rule is worthless no matter how high its confidence · < 1: negative association.", "glossary": []},
        {"id": "f4", "name": "Coverage", "latex": r"\text{coverage}(X \Rightarrow Y) = \text{supp}(X)",
         "caption": "What fraction of clients the rule even applies to. A rule with lift 4 that fires on 0.1% of clients is a curiosity, not a policy.", "glossary": []},
        {"id": "f5", "name": "Leverage and conviction",
         "latex": r"\text{leverage} = \text{supp}(X \cup Y) - \text{supp}(X)\,\text{supp}(Y), \qquad \text{conviction} = \frac{1 - \text{supp}(Y)}{1 - \text{conf}(X \Rightarrow Y)}",
         "caption": "Leverage: how many more co-occurrences than independence predicts (as a fraction of all clients). Conviction: how much more often the rule would be wrong if X and Y were independent.", "glossary": []},
        {"id": "f6", "name": "Apriori property (downward closure)", "latex": r"X \text{ infrequent} \;\Longrightarrow\; \forall\, Y \supseteq X,\; Y \text{ infrequent}",
         "caption": "If an itemset is rare, every superset is at least as rare. This is what makes the search tractable — without it you would test 2^m − 1 itemsets.", "glossary": []},
    ],
    "assumptions": [
        {"text": "Items are discrete", "status": "warn", "evidence": "Continuous features were binned; bin edges (decades, terciles) are choices, and the macro terciles are uneven because those features take few distinct values."},
        {"text": "Minimum support filters noise", "status": "ok", "evidence": "1% = 330 training clients; education=illiterate (18 rows) cannot appear in any rule."},
        {"text": "Confidence is meaningful on its own", "status": "fail", "evidence": "88.73% of clients say no, so any rule → y=no has ~0.89 confidence by default. Rank by lift."},
        {"text": "Rules generalise beyond the mining data", "status": "unknown", "evidence": "Checked: rules mined on the training split are re-measured on the held-out test split."},
    ],
}


def main():
    df = clean(load_raw())
    y = df[C.TARGET].to_numpy()
    tr, te = split_indices(len(df), y)
    B_all = make_baskets(df)
    B, B_test = B_all.iloc[tr].reset_index(drop=True), B_all.iloc[te].reset_index(drop=True)
    n = len(B)
    print(f"FP-Growth: {n:,} training baskets × {B.shape[1]} items", flush=True)

    # Runtime: Apriori vs FP-Growth at identical min_support (and our own Apriori)
    runtime = []
    for s in RUNTIME_SUPPORTS:
        row = {"min_support": s}
        for name, fn, kw in [("fpgrowth", fpgrowth, {}), ("apriori", apriori, {"low_memory": True})]:
            t0 = time.perf_counter()
            res = fn(B, min_support=s, use_colnames=True, max_len=MAX_LEN, **kw)
            row[name] = time.perf_counter() - t0
            row[f"{name}_itemsets"] = len(res)
        t0 = time.perf_counter()
        own_apriori(B, s, MAX_LEN)
        row["own_apriori"] = time.perf_counter() - t0
        runtime.append(row)
        print(f"  min_support={s}: fpgrowth {row['fpgrowth']:.2f}s, apriori {row['apriori']:.2f}s, ours {row['own_apriori']:.2f}s, "
              f"{row['fpgrowth_itemsets']:,} itemsets", flush=True)
    assert all(r["fpgrowth_itemsets"] == r["apriori_itemsets"] for r in runtime), "Apriori and FP-Growth disagree"

    levels = own_apriori(B, MIN_SUPPORT, MAX_LEN)
    freq = fpgrowth(B, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN)
    by_len = freq["itemsets"].map(len).value_counts().sort_index()
    assert all(lv["frequent"] == int(by_len.get(lv["k"], 0)) for lv in levels), "own Apriori disagrees with mlxtend"
    rules = association_rules(freq, num_itemsets=n, metric="confidence", min_threshold=MIN_CONF)
    rules = rules.replace([np.inf], np.nan)
    yes = rules[rules.consequents.map(lambda c: c == frozenset({"y=yes"}))].sort_values("lift", ascending=False)
    no_rules = rules[rules.consequents.map(lambda c: c == frozenset({"y=no"}))]
    actionable_all = yes[yes.lift >= MIN_LIFT]
    # Minimum improvement: drop "poutcome=success + loan=no → yes" when "poutcome=success → yes" is as good.
    supp = dict(zip(freq.itemsets, freq.support))
    base_conf = float(B["y=yes"].mean())

    def improvement(ante):
        conf = supp[ante | {"y=yes"}] / supp[ante]
        subs = [frozenset(c) for k in range(1, len(ante)) for c in itertools.combinations(ante, k)]
        best_sub = max([base_conf] + [supp[s_ | {"y=yes"}] / supp[s_] for s_ in subs])
        return conf - best_sub

    imp = actionable_all.antecedents.map(improvement)
    actionable = actionable_all[imp >= MIN_IMPROVEMENT].assign(improvement=imp[imp >= MIN_IMPROVEMENT])
    print(f"  minimum-improvement filter: {len(actionable_all)} → {len(actionable)} rules")
    print(f"  {len(freq):,} frequent itemsets, {len(rules):,} rules, {len(actionable_all)} with lift ≥ {MIN_LIFT}, {len(actionable)} after the improvement filter")

    # Re-measure on the held-out test split; and use the rules as a scorer
    top = []
    for _, r in actionable.head(40).iterrows():
        rec = rule_record(r, n)
        rec["test_confidence"], rec["test_lift"], rec["test_matches"] = test_stats(B_test, rec["antecedent"], rec["consequent"])
        top.append(rec)
    score = np.full(len(B_test), y[tr].mean())
    for _, r in actionable_all.iterrows():
        m = B_test[list(r["antecedents"])].all(axis=1).to_numpy()
        score[m] = np.maximum(score[m], r["confidence"])
    rule_auc = float(roc_auc_score(y[te], score))
    write_json(model_dir(MID) / "predictions.json", {"test_index": te.tolist(), "y_true": y[te].tolist(),
                                                     "score_name": "confidence of the best actionable rule the client matches",
                                                     "without_duration": {"proba": np.round(score, 5).tolist()}})

    # The confidence trap: a high-confidence, lift≈1 rule → y=no beside a useful lower-confidence rule
    cand = no_rules.assign(dist=(no_rules.lift - 1).abs(), size=no_rules.antecedents.map(len)).query("confidence > 0.85 and dist < 0.01")
    trap = cand.sort_values(["size", "support"], ascending=[True, False]).iloc[0]
    useful = yes[yes.antecedents.map(len) == 1].sort_values("lift", ascending=False).iloc[0]
    base_yes = float(B["y=yes"].mean())

    # Plan's expected rules (§7.6) — verified, not copied
    def single(ante):
        sub = B[list(ante)].all(axis=1)
        conf = B.loc[sub, "y=yes"].mean()
        return {"rule": f"{fs(ante)} → y=yes", "support": float((sub & B['y=yes']).mean()), "coverage": float(sub.mean()),
                "confidence": float(conf), "lift": float(conf / base_yes)}
    checks = [single(["poutcome=success"]), single(["job=student"]), single(["job=retired"])]
    for mo in ["mar", "sep", "oct", "dec"]:
        checks.append(single(["contact=cellular", f"month={mo}"]))

    d = model_dir(MID)
    figs = []

    # 1 — rule network
    net = actionable.head(20)
    items = sorted({i for a in net.antecedents for i in a})
    ang = np.linspace(0, 2 * np.pi, len(items), endpoint=False)
    ipos = {it: (np.cos(a) * 2.2, np.sin(a) * 2.2) for it, a in zip(items, ang)}
    fig = go.Figure()
    lifts = net.lift.to_numpy()
    for k, (_, r) in enumerate(net.iterrows()):
        a = 2 * np.pi * k / len(net)
        rx, ry = np.cos(a) * 1.1, np.sin(a) * 1.1
        w = 0.6 + 3.5 * (r.lift - lifts.min()) / max(lifts.max() - lifts.min(), 1e-9)
        for it in r.antecedents:
            fig.add_trace(go.Scatter(x=[ipos[it][0], rx], y=[ipos[it][1], ry], mode="lines", line=dict(color=BLUE, width=0.8),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[rx, 0], y=[ry, 0], mode="lines", line=dict(color=ACCENT, width=w), opacity=0.7,
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[rx], y=[ry], mode="markers", marker=dict(size=7, color=AMBER), showlegend=False,
                                 hovertemplate=f"{fs(r.antecedents)} → y=yes<br>lift {r.lift:.2f} · conf {r.confidence:.2f} · "
                                               f"coverage {r['antecedent support']:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[p[0] for p in ipos.values()], y=[p[1] for p in ipos.values()], mode="markers+text",
                             text=list(ipos), textposition="top center", marker=dict(size=10, color=BLUE), showlegend=False,
                             hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", text=["y=yes"], textposition="bottom center",
                             marker=dict(size=22, color=ACCENT), showlegend=False, hoverinfo="skip"))
    fig.update_layout(title="Top 20 rules → y=yes (edge width = lift; amber dots = rules)", xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x"), height=620)
    write_figure(MID, "rule_network.json", fig)
    figs.append({"file": "rule_network.json", "title": "Rule network", "type": "network",
                 "caption": "Items on the outer ring, rules (amber) in the middle, the consequent y=yes at the centre. Thicker edge = higher lift. Hover a rule for its measures."})

    # 2 — support vs confidence scatter coloured by lift
    rr = rules.sample(min(4000, len(rules)), random_state=C.RANDOM_STATE)
    fig = go.Figure(go.Scattergl(x=rr.support, y=rr.confidence, mode="markers",
                                 marker=dict(size=4, color=rr.lift, colorscale=[[0, "#C88B8B"], [0.25, "#E8E3E0"], [1, ACCENT]],
                                             cmin=0, cmax=min(6, rr.lift.max()), colorbar=dict(title="lift")),
                                 text=[f"{fs(a)} → {fs(c)}<br>lift {l:.2f}" for a, c, l in zip(rr.antecedents, rr.consequents, rr.lift)],
                                 hovertemplate="%{text}<br>support %{x:.3f} · confidence %{y:.2f}<extra></extra>"))
    for r, nm, col in [(trap, "the trap: high confidence, lift ≈ 1", ROSE), (useful, "useful: lower confidence, high lift", ACCENT)]:
        fig.add_annotation(x=r.support, y=r.confidence, text=nm, showarrow=True, arrowhead=2, ax=60, ay=-40, font=dict(color=col))
    fig.update_layout(title=f"Support vs confidence for {len(rr):,} rules (colour = lift)", xaxis_title="support", xaxis_type="log",
                      yaxis_title="confidence")
    write_figure(MID, "support_confidence.json", fig)
    figs.append({"file": "support_confidence.json", "title": "Support vs confidence", "type": "scatter",
                 "caption": "The top band (confidence ≈ 0.9) is almost entirely rules predicting y=no with lift ≈ 1 — they restate the base rate. The useful rules sit lower, in teal."})

    # 3 — top rules table
    write_json(d / "figures" / "top_rules.json", {
        "type": "table", "columns": ["rule", "support", "coverage", "confidence", "lift", "leverage", "conviction", "test_confidence", "test_lift"],
        "format": {k: "pct" if k in ("support", "coverage", "confidence", "test_confidence") else "num" for k in
                   ["support", "coverage", "confidence", "lift", "leverage", "conviction", "test_confidence", "test_lift"]},
        "rows": [{"rule": f"{fs(r['antecedent'])} → {fs(r['consequent'])}", **{k: r[k] for k in
                  ["support", "coverage", "confidence", "lift", "leverage", "conviction", "test_confidence", "test_lift"]}} for r in top[:25]]})
    figs.append({"file": "top_rules.json", "title": "Top rules by lift (→ y=yes)", "type": "table",
                 "caption": "All six measures, mined on the training split; the last two columns re-measure each rule on held-out test clients."})

    # 4 — L1–L4 progression
    ks = [f"L{lv['k']}" for lv in levels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ks, y=[lv["candidates"] for lv in levels], name="candidates generated", marker_color=MUTED))
    fig.add_trace(go.Bar(x=ks, y=[lv["pruned_by_subset"] for lv in levels], name="pruned by downward closure (f6)", marker_color=AMBER))
    fig.add_trace(go.Bar(x=ks, y=[lv["pruned_by_support"] for lv in levels], name="pruned by support count", marker_color=ROSE))
    fig.add_trace(go.Bar(x=ks, y=[lv["frequent"] for lv in levels], name="frequent", marker_color=ACCENT))
    fig.update_layout(barmode="group", title=f"Apriori level by level (min_support = {MIN_SUPPORT:.0%})", yaxis_type="log",
                      yaxis_title="itemsets (log scale)", legend=dict(orientation="h", y=-0.2))
    write_figure(MID, "levels.json", fig)
    figs.append({"file": "levels.json", "title": "L1 → L4 itemset progression", "type": "bar",
                 "caption": " · ".join(f"L{lv['k']}: {lv['candidates']:,} candidates → {lv['frequent']:,} frequent" for lv in levels)
                            + ". Counts from our own Apriori, cross-checked against mlxtend."})

    # 5 — runtime
    fig = go.Figure()
    xs = [f"{r['min_support']:.0%}" for r in runtime]
    fig.add_trace(go.Bar(x=xs, y=[r["apriori"] for r in runtime], name="Apriori (mlxtend)", marker_color=AMBER))
    fig.add_trace(go.Bar(x=xs, y=[r["fpgrowth"] for r in runtime], name="FP-Growth (mlxtend)", marker_color=ACCENT))
    fig.add_trace(go.Bar(x=xs, y=[r["own_apriori"] for r in runtime], name="Apriori (ours, bit-packed)", marker_color=BLUE))
    fig.update_layout(barmode="group", title=f"Runtime at identical min_support (max itemset size {MAX_LEN}, {n:,} baskets)",
                      xaxis_title="min_support", yaxis_title="seconds", legend=dict(orientation="h", y=-0.2))
    write_figure(MID, "runtime.json", fig)
    last = runtime[-1]
    figs.append({"file": "runtime.json", "title": "Apriori vs FP-Growth runtime", "type": "bar",
                 "caption": f"At min_support 1%: FP-Growth {last['fpgrowth']:.1f}s vs Apriori {last['apriori']:.1f}s vs ours {last['own_apriori']:.1f}s for the same "
                            f"{last['fpgrowth_itemsets']:,} itemsets (asserted identical). mlxtend's default Apriori ran out of memory at 1%; low_memory mode was used throughout."})

    trap_rec, useful_rec = rule_record(trap, n), rule_record(useful, n)
    write_json(d / "confidence_trap.json", {
        "type": "confidence_trap", "base_rate_no": float(B["y=no"].mean()), "base_rate_yes": base_yes,
        "trap": trap_rec, "useful": useful_rec,
        "explanation": f"The first rule is right {trap_rec['confidence']:.1%} of the time — but {B['y=no'].mean():.1%} of all clients say no anyway, "
                       f"so its lift is {trap_rec['lift']:.3f}: knowing the antecedent tells you nothing. The second rule is right only "
                       f"{useful_rec['confidence']:.1%} of the time, yet that is {useful_rec['lift']:.1f}× the {base_yes:.1%} base rate. "
                       "Confidence ignores how common the consequent already is; lift does not."})

    write_json(d / "meta.json", {**META, "figures": figs,
                                 "extras": [{"kind": "confidence_trap", "file": "confidence_trap.json", "title": "The confidence trap"}]})
    write_json(d / "config.json", {"estimator": "mlxtend.frequent_patterns.fpgrowth + association_rules", "params": [
        {"name": "min_support", "value": MIN_SUPPORT, "why": "1% of training clients = 330 people: the smallest group worth a targeted call list."},
        {"name": "min_confidence", "value": MIN_CONF, "why": "Rules must hold at least 30% of the time — ~2.7× the 11.27% base rate for y=yes."},
        {"name": "min_lift", "value": MIN_LIFT, "why": "Reporting filter: a rule must lift the subscription rate by at least 20% over chance."},
        {"name": "min_improvement", "value": MIN_IMPROVEMENT, "why": "Redundancy filter (Bayardo, Agrawal & Gunopulos 1999): adding an item must raise confidence by ≥ 2 points over every simpler rule, otherwise the longer rule is just the shorter one restated."},
        {"name": "max_len", "value": MAX_LEN, "why": "Itemsets up to 4 items (the L1–L4 progression); longer rules have tiny coverage and are hard to act on."},
        {"name": "binning", "value": "age decades; campaign {1, 2–3, 4+}; macro terciles", "why": "Rules need discrete items. Terciles keep each macro bin around a third of clients where ties allow."},
        {"name": "mining data", "value": "training split only", "why": "So rules can be re-checked on held-out clients, like every other model."},
    ]})
    write_json(d / "metrics.json", {"model_id": MID, "task": "association_rules", "baskets": n, "items": int(B.shape[1]),
                                    "frequent_itemsets": int(len(freq)), "levels": levels, "rules_total": int(len(rules)),
                                    "rules_to_yes": int(len(yes)), "rules_actionable": int(len(actionable)),
                                    "rules_before_improvement_filter": int(len(actionable_all)), "min_improvement": MIN_IMPROVEMENT,
                                    "rule_score_auc_test": rule_auc, "runtime": runtime, "expected_rules_check": checks})

    # Explain panel: rule contributions per item
    item_rows = {}
    for _, r in actionable.iterrows():
        for it in r.antecedents:
            item_rows.setdefault(it, []).append(r)
    items_payload = []
    for it, rs in sorted(item_rows.items(), key=lambda kv: -max(r.lift for r in kv[1]))[:20]:
        rs = sorted(rs, key=lambda r: -r.lift)[:5]
        items_payload.append({"item": it, "rules": [{"rule": f"{fs(r.antecedents)} → y=yes", "lift": float(r.lift),
                                                     "confidence": float(r.confidence), "support": float(r.support),
                                                     "coverage": float(r["antecedent support"])} for r in rs],
                              "plain_text": f"{it} appears in {len(item_rows[it])} actionable rules; the strongest lifts subscription "
                                            f"{rs[0].lift:.1f}× over the base rate, covering {rs[0]['antecedent support']:.1%} of clients."})
    best = top[0]
    write_json(d / "explain.json", {
        "kind": "rule_contribution", "items": items_payload,
        "global": {"plain_text": f"{len(actionable)} rules predict subscription with lift ≥ {MIN_LIFT}. The strongest items are "
                                 + ", ".join(p["item"] for p in items_payload[:4]) + "."},
        "narrative": {
            "headline": f"The best rule lifts subscription {best['lift']:.1f}×: {fs(best['antecedent'])} → y=yes.",
            "body": f"poutcome=success alone lifts subscription {checks[0]['lift']:.2f}× (the plan expected ≈ 5.8). Rules also find the "
                    f"low-interest-rate period and cellular contact. Used as a scorer, the rules rank held-out clients at AUC {rule_auc:.3f}.",
            "caveat": "SHAP does not apply to rule mining. A rule describes co-occurrence in this data, not a cause — and a high-confidence rule can be useless (see the confidence trap)."}})

    write_json(d / "findings.json", {
        "headline": f"Rules that beat chance exist — poutcome=success lifts subscription {checks[0]['lift']:.1f}× — but the highest-confidence rules are worthless.",
        "findings": [
            {"type": "result", "text": f"{len(freq):,} frequent itemsets (≤ {MAX_LEN} items) and {len(rules):,} rules at confidence ≥ {MIN_CONF}; "
                                       f"{len(yes)} predict y=yes, {len(actionable_all)} of them with lift ≥ {MIN_LIFT}."},
            {"type": "result", "text": f"Of {len(actionable_all)} rules with lift ≥ {MIN_LIFT}, only {len(actionable)} survive the minimum-improvement filter: most long rules are "
                                       "a strong short rule (usually poutcome=success) plus an item that adds nothing. Ranking by lift alone fills the top of the list with such restatements."},
            {"type": "result", "text": "Plan's expected rules, verified on training data: " + "; ".join(
                f"{c['rule']} lift {c['lift']:.2f} (coverage {c['coverage']:.1%})" for c in checks) + "."},
            {"type": "result", "text": f"Rules generalise: the top rule's confidence is {best['confidence']:.2f} on training clients and "
                                       f"{best['test_confidence']:.2f} on {best['test_matches']} held-out clients."},
            {"type": "result", "text": speed_sentence(runtime)},
            {"type": "failure", "text": "mlxtend's default Apriori ran out of memory at 1% support: its level-4 step tried to allocate a 32,950 × 96,363 boolean matrix (2.96 GiB). We re-ran every support level with low_memory=True — candidate explosion, observed directly."},
            {"type": "result", "text": f"Implementation matters as much as algorithm: our bit-packed Apriori (rows packed into 64-bit words, support = popcount) took {last['own_apriori']:.1f}s at 1% — faster than both library implementations. FP-Growth's advantage is over naive candidate counting, not over every Apriori."},
            {"type": "limitation", "text": f"The confidence trap: {fs(trap_rec['antecedent'])} → y=no has confidence {trap_rec['confidence']:.3f} and lift {trap_rec['lift']:.3f}. It restates the base rate."},
            {"type": "limitation", "text": f"As a classifier the rules are weak (AUC {rule_auc:.3f}): most clients match no actionable rule, so they all get the same score."},
            {"type": "limitation", "text": "Rules are only as good as the bins: the macro terciles are uneven because those features take few distinct values."},
        ],
        "comparison_note": "The only method whose output is a readable policy ('call these people'), but it describes groups, not individuals, and ignores everyone outside the rules' coverage.",
        "viva_answer": "Q: You have a rule with 89% confidence. Is it a good rule? — Almost certainly not. 88.73% of all clients decline, so a rule predicting 'no' with 89% confidence is barely better than the base rate — its lift is about 1.0, meaning the antecedent tells you nothing. Confidence ignores how common the consequent already is. That is why we rank by lift and report coverage alongside it.",
    })
    validate_model_artifacts(MID)
    print(f"fp_growth artifacts valid; rule AUC {rule_auc:.3f}; checks {[(c['rule'], round(c['lift'], 2)) for c in checks]}")


if __name__ == "__main__":
    main()
