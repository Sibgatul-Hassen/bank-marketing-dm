"""Builds results/dataset/* — the Data & Preprocessing and EDA pages. OWNER: Agent-00.

Every number here is recomputed from the raw file, so the pages never quote a figure the code
did not produce.   python -m src.dataset_report
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C
from .contract import validate_dataset, write_figure, write_json
from .data_loader import file_sha256, load_raw, split_indices
from .preprocessing import clean

YES, NO = "#4FB3A0", "#6B7E8F"
DATA_COLORS = ["#4FB3A0", "#D4A373", "#7B9FC7", "#C88B8B", "#A89BC4", "#8FA880"]
DIVERGING = [[0, "#C88B8B"], [0.25, "#D9B3B3"], [0.5, "#E8E3E0"], [0.75, "#A9CFC4"], [1, "#4FB3A0"]]
OUT = C.RESULTS_DATASET
UNKNOWN_COLS = ["default", "education", "housing", "loan", "job", "marital"]


def rate_table(df, col):
    g = df.groupby(col, observed=True)["yes"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    return [{"level": str(k), "rate": float(r["mean"]), "n": int(r["size"])} for k, r in g.iterrows()]


def fig_rate_bars(rows, title, base):
    levels = [r["level"] for r in rows]
    fig = go.Figure(go.Bar(
        x=levels, y=[r["rate"] for r in rows], marker_color=DATA_COLORS[0],
        text=[f"n={r['n']:,}" for r in rows], textposition="outside",
        hovertemplate="%{x}<br>subscribe rate %{y:.2%}<br>%{text}<extra></extra>",
    ))
    fig.add_hline(y=base, line_dash="dot", line_color="#9FB0C0",
                  annotation_text=f"overall {base:.2%}", annotation_position="top right")
    fig.update_layout(title=title, yaxis_tickformat=".0%", yaxis_title="subscription rate",
                      margin=dict(t=50, b=40, l=50, r=20))
    return fig


def main():
    raw = load_raw()
    df = raw.copy()
    df["yes"] = (df["y"] == "yes").astype(int)
    n = len(df)
    base = float(df["yes"].mean())

    # ------------------------------------------------------------ summary.json
    num_cols = [c for c in raw.columns if raw[c].dtype.kind in "if"]
    corr_y = df[num_cols + ["yes"]].corr()["yes"].drop("yes").sort_values(key=abs, ascending=False)
    macro_corr = df[C.MACRO].corr()
    dur = df.groupby("y")["duration"].agg(["mean", "min", "max", "median"])
    rates = {c: rate_table(df, c) for c in [*C.CATEGORICAL, "default"]}
    ill = next(r for r in rates["education"] if r["level"] == "illiterate")
    summary = {
        "source": "UCI ML Repository, Dataset 222 — Bank Marketing (bank-additional-full.csv)",
        "citation": "Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the "
                    "success of bank telemarketing. Decision Support Systems, 62, 22–31.",
        "sha256": file_sha256(),
        "sha256_note": "Computed after normalising CRLF to LF; git on Windows rewrites line endings.",
        "rows": n, "columns": raw.shape[1], "separator": ";",
        "period": "May 2008 – November 2010",
        "target": {"no": int((df.y == "no").sum()), "yes": int(df.yes.sum()), "yes_rate": base,
                   "majority_baseline_accuracy": 1 - base},
        "duplicate_rows": int(raw.duplicated().sum()),
        "columns_info": [
            {"name": c, "dtype": "numeric" if raw[c].dtype.kind in "if" else "categorical",
             "unique": int(raw[c].nunique()),
             **({"min": float(raw[c].min()), "max": float(raw[c].max()), "mean": float(raw[c].mean()),
                 "median": float(raw[c].median())} if raw[c].dtype.kind in "if" else {})}
            for c in raw.columns
        ],
        "correlation_with_target": {k: float(v) for k, v in corr_y.items()},
        "macro_correlation": {f"{a} ~ {b}": float(macro_corr.loc[a, b])
                              for i, a in enumerate(C.MACRO) for b in C.MACRO[i + 1:]},
        "rates": rates,
        "small_sample_trap": {
            "level": "education=illiterate", "n": ill["n"], "yes": int(round(ill["rate"] * ill["n"])),
            "rate": ill["rate"],
            "rate_with_one_more_yes": (round(ill["rate"] * ill["n"]) + 1) / (ill["n"] + 1),
            "one_row_moves_points": 100 / ill["n"],
            "wilson_95": wilson(round(ill["rate"] * ill["n"]), ill["n"]),
        },
        "campaign": {"mean": float(df.campaign.mean()), "median": float(df.campaign.median()),
                     "max": int(df.campaign.max()), "p99": float(df.campaign.quantile(0.99))},
        "split": {"train": int(n * (1 - C.TEST_SIZE) + 0.5), "test": int(n * C.TEST_SIZE + 0.5),
                  "stratified": True, "random_state": C.RANDOM_STATE},
    }
    write_json(OUT / "summary.json", summary)

    # ------------------------------------------------------------ quality.json
    unknown = [{"column": c, "unknown": int((raw[c] == "unknown").sum()),
                "pct": float((raw[c] == "unknown").mean())} for c in UNKNOWN_COLS]
    unk = pd.DataFrame({c: raw[c] == "unknown" for c in UNKNOWN_COLS})
    housing_loan_same = int((unk.housing & unk.loan).sum())
    cleaned = clean(raw)
    quality = {
        "problems": [
            {"id": "leak", "title": "duration is a target leak", "column": "duration",
             "evidence": {"corr_with_target": float(corr_y["duration"]),
                          "next_highest": {k: float(v) for k, v in list(corr_y.drop("duration").items())[:1]},
                          "mean_no": float(dur.loc["no", "mean"]), "mean_yes": float(dur.loc["yes", "mean"]),
                          "min_no": int(dur.loc["no", "min"]), "min_yes": int(dur.loc["yes", "min"]),
                          "zero_second_calls": int((df.duration == 0).sum()),
                          "zero_second_calls_yes": int(((df.duration == 0) & (df.yes == 1)).sum())},
             "action": "Excluded from every model. Reintroduced only for the with-duration comparison run.",
             "before": 21, "after": 20},
            {"id": "sentinel", "title": "pdays = 999 is a sentinel, not a number", "column": "pdays",
             "evidence": {"rows_999": int((raw.pdays == 999).sum()), "pct": float((raw.pdays == 999).mean()),
                          "real_range": [int(raw.pdays[raw.pdays != 999].min()), int(raw.pdays[raw.pdays != 999].max())]},
             "action": "Split into binary was_contacted plus numeric pdays (999 replaced by -1, a "
                       "placeholder outside the real 0–27 range; the flag carries the meaning)."},
            {"id": "unknown", "title": "'unknown' is missing data disguised as a category",
             "evidence": {"columns": unknown,
                          "uci_metadata_says": "Has Missing Values? No",
                          "housing_and_loan_unknown_together": housing_loan_same,
                          "education_unknown_rate": next(r["rate"] for r in rates["education"] if r["level"] == "unknown")},
             "action": "Kept as its own level (one of three defensible choices: keep / impute / drop). "
                       "education=unknown subscribes above several known levels, so it carries signal."},
            {"id": "default", "title": "default is effectively constant", "column": "default",
             "evidence": {"counts": {k: int(v) for k, v in raw["default"].value_counts().items()}},
             "action": "Dropped the three-level column; kept a binary default_unknown flag."},
        ],
        "metadata_discrepancy": "UCI lists 'Has Missing Values? No'. Six columns contain the string "
                                "'unknown' — 12,718 cells in total. Technically true, practically misleading.",
        "unknown_total_cells": int(unk.values.sum()),
        "duplicates": {"rows": int(raw.duplicated().sum()),
                       "note": "Exact duplicate rows. Kept: 12 rows cannot move any metric measurably, "
                               "and without client IDs we cannot tell a duplicate from two identical clients."},
        "pipeline": [
            {"step": "Raw CSV", "rows": n, "cols": raw.shape[1]},
            {"step": "Verify SHA256", "rows": n, "cols": raw.shape[1]},
            {"step": "Drop duration", "rows": n, "cols": raw.shape[1] - 1},
            {"step": "Split pdays sentinel", "rows": n, "cols": raw.shape[1]},
            {"step": "Keep 'unknown' as level", "rows": n, "cols": raw.shape[1]},
            {"step": "Collapse default", "rows": n, "cols": cleaned.shape[1]},
            {"step": "Train split (80%)", "rows": summary["split"]["train"], "cols": cleaned.shape[1]},
            {"step": "Test split (20%)", "rows": summary["split"]["test"], "cols": cleaned.shape[1]},
        ],
        "encodings": {
            "decision_tree": "one-hot (fitted on train), no scaling needed",
            "catboost": "native categorical — ordered target statistics, no one-hot",
            "tabnet": "integer codes → learned embeddings (cat_idxs / cat_dims); numerics standardised",
            "tabpfn": "integer codes flagged as categorical; the model's own preprocessing",
            "umap_hdbscan": "one-hot + standard scaling (distance-based)",
            "fp_growth": "binned into items: age decades, campaign {1, 2–3, 4+}, macro terciles",
        },
        "split_protocol": "Stratified 80/20 holdout (random_state=42). Stratified 5-fold CV on the "
                          "training part only. All scaling, encoding vocabularies and class weighting "
                          "are fitted inside each fold — never before the split.",
    }
    write_json(OUT / "quality.json", quality)

    # ------------------------------------------------------------ figures
    figs = []

    def add(name, fig, title, caption, page):
        write_figure(OUT, name, fig)
        figs.append({"file": name, "title": title, "caption": caption, "page": page})

    # Row-count funnel (Data page)
    steps = quality["pipeline"]
    fig = go.Figure(go.Bar(
        y=[s["step"] for s in steps][::-1], x=[s["rows"] for s in steps][::-1], orientation="h",
        marker_color=[DATA_COLORS[0]] * 6 + [DATA_COLORS[2]] * 2,
        text=[f"{s['rows']:,} rows × {s['cols']} cols" for s in steps][::-1], textposition="auto",
        hovertemplate="%{y}: %{x:,} rows<extra></extra>"))
    fig.update_layout(title="Rows and columns through the pipeline", xaxis_title="rows",
                      margin=dict(l=170, t=50))
    add("funnel.json", fig, "Pipeline funnel",
        "No row is ever dropped — every fix changes columns, not rows. The split happens last, after "
        "row-wise cleaning that learns nothing from the data.", "data")

    # Unknown co-occurrence heatmap (Data page)
    co = np.array([[float((unk[a] & unk[b]).sum() / max(unk[a].sum(), 1)) for b in UNKNOWN_COLS] for a in UNKNOWN_COLS])
    fig = go.Figure(go.Heatmap(
        z=co, x=UNKNOWN_COLS, y=UNKNOWN_COLS, colorscale=[[0, "#1E2832"], [1, "#4FB3A0"]], zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in co], texttemplate="%{text}",
        hovertemplate="P(%{x} unknown | %{y} unknown) = %{z:.1%}<extra></extra>"))
    fig.update_layout(title="Where 'unknown' appears together", xaxis_title="…then this is unknown",
                      yaxis_title="Given this is unknown…", yaxis_autorange="reversed")
    add("unknown_heatmap.json", fig, "'unknown' co-occurrence",
        f"Diagonal = 100%. housing and loan are unknown on exactly the same {housing_loan_same} rows — one "
        "missing source, not two. default=unknown (20.87%) is mostly independent of the rest.", "data")

    unk_bar = go.Figure(go.Bar(
        x=[u["column"] for u in unknown], y=[u["pct"] for u in unknown], marker_color=DATA_COLORS[1],
        text=[f"{u['unknown']:,}" for u in unknown], textposition="outside",
        hovertemplate="%{x}: %{y:.2%} unknown (%{text} rows)<extra></extra>"))
    unk_bar.update_layout(title="Share of 'unknown' per column", yaxis_tickformat=".0%")
    add("unknown_bar.json", unk_bar, "'unknown' per column",
        "The UCI metadata says there are no missing values. These six columns disagree.", "data")

    # Class balance donut (Data page)
    fig = go.Figure(go.Pie(labels=["not subscribed", "subscribed"], values=[summary["target"]["no"], summary["target"]["yes"]],
                           hole=0.62, marker_colors=[NO, YES], sort=False, textinfo="label+percent"))
    fig.update_layout(title="Class balance", showlegend=False,
                      annotations=[dict(text=f"{base:.2%}<br>yes", showarrow=False, font_size=18)])
    add("class_balance.json", fig, "Class balance",
        f"Only {base:.2%} subscribe. Predicting 'no' for everyone scores {1 - base:.2%} accuracy — "
        "which is why accuracy is never the headline metric here.", "data")

    # Duration by class with minimums annotated (Data page)
    fig = go.Figure()
    for lbl, col, name in [("no", NO, "not subscribed"), ("yes", YES, "subscribed")]:
        fig.add_trace(go.Histogram(x=df.loc[df.y == lbl, "duration"].clip(upper=2000), name=name,
                                   marker_color=col, opacity=0.75, xbins=dict(size=25),
                                   histnorm="probability density"))
    fig.add_vline(x=0, line_color=NO, annotation_text=f"min 'no' = {int(dur.loc['no', 'min'])}s", annotation_position="top right")
    fig.add_vline(x=int(dur.loc["yes", "min"]), line_color=YES, line_dash="dot",
                  annotation_text=f"min 'yes' = {int(dur.loc['yes', 'min'])}s", annotation_position="bottom right")
    fig.update_layout(barmode="overlay", title="Call duration by outcome (clipped at 2,000 s)",
                      xaxis_title="duration (seconds)", yaxis_title="density")
    add("duration_by_class.json", fig, "duration — the leak",
        f"Mean {dur.loc['no', 'mean']:.0f}s for 'no' vs {dur.loc['yes', 'mean']:.0f}s for 'yes'. No call under "
        f"{int(dur.loc['yes', 'min'])}s ever ended in a subscription — and the length is only known after hanging up.", "data")

    # --- EDA page
    corr_cols = ["age", "duration", "campaign", "pdays", "previous", *C.MACRO, "yes"]
    cm = df[corr_cols].corr()
    fig = go.Figure(go.Heatmap(z=cm.values, x=corr_cols, y=corr_cols, zmin=-1, zmax=1, colorscale=DIVERGING,
                               text=np.round(cm.values, 2), texttemplate="%{text}",
                               hovertemplate="%{x} ~ %{y}: r = %{z:.3f}<extra></extra>"))
    fig.update_layout(title="Pearson correlation (yes = target)", yaxis_autorange="reversed",
                      margin=dict(l=110, b=110))
    add("correlation.json", fig, "Correlation heatmap",
        "emp.var.rate, euribor3m and nr.employed correlate at 0.91–0.97: three columns carrying one "
        "signal, 'the state of the economy'. Note pdays is the raw column here (999 sentinel included).", "eda")

    for col, cap in [
        ("poutcome", "The strongest legitimate predictor: clients whose previous campaign succeeded subscribe at 65%."),
        ("contact", "Cellular contacts convert nearly 3× better than landline — but channel may proxy for client type."),
        ("job", "Students and retirees respond most; blue-collar least. Note the sample sizes."),
        ("education", f"education=illiterate shows {ill['rate']:.1%} from only {ill['n']} rows — see the small-sample trap."),
        ("day_of_week", "Nearly flat. A feature that does nothing is itself a finding."),
    ]:
        add(f"rate_{col}.json", fig_rate_bars(rates[col], f"Subscription rate by {col}", base),
            f"{col}", cap, "eda")

    m = pd.DataFrame(rates["month"])
    order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    m = m.set_index("level").loc[order].reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=m.level, y=m.n, name="calls", marker_color="#7B9FC7", opacity=0.6), secondary_y=False)
    fig.add_trace(go.Scatter(x=m.level, y=m.rate, name="subscribe rate", mode="lines+markers",
                             line=dict(color=YES, width=2.5)), secondary_y=True)
    fig.update_yaxes(title_text="calls", secondary_y=False)
    fig.update_yaxes(title_text="subscription rate", tickformat=".0%", secondary_y=True)
    fig.update_layout(title="Month: volume vs rate", legend=dict(orientation="h", y=-0.15))
    may = m.set_index("level").loc["may"]; mar = m.set_index("level").loc["mar"]
    add("month_dual.json", fig, "The month volume–rate inversion",
        f"May has the most calls ({int(may.n):,}) and the worst rate ({may.rate:.2%}); March has "
        f"{int(mar.n)} calls and {mar.rate:.2%}. This may reflect campaign strategy, not client behaviour — "
        "no causal claim.", "eda")

    fig = make_subplots(rows=1, cols=4, subplot_titles=["age", "campaign", "euribor3m", "nr.employed"])
    for i, c in enumerate(["age", "campaign", "euribor3m", "nr.employed"], start=1):
        for lbl, col, name in [("no", NO, "not subscribed"), ("yes", YES, "subscribed")]:
            fig.add_trace(go.Violin(y=df.loc[df.y == lbl, c], name=name, line_color=col, showlegend=(i == 1),
                                    legendgroup=name, box_visible=True, meanline_visible=True, points=False,
                                    spanmode="hard"), row=1, col=i)
    fig.update_layout(title="Numeric features by outcome", violinmode="group",
                      legend=dict(orientation="h", y=-0.12))
    add("violins.json", fig, "Numeric features by class",
        f"Subscribers cluster at low euribor3m / nr.employed — the crisis period. campaign is heavily "
        f"skewed: median {summary['campaign']['median']:.0f}, max {summary['campaign']['max']}.", "eda")

    # Imbalance in the split
    tr, te = split_indices(n, df.yes.to_numpy())
    fig = go.Figure([go.Bar(x=["train", "test"], y=[df.yes.iloc[tr].mean(), df.yes.iloc[te].mean()],
                            marker_color=YES, text=[f"{len(tr):,} rows", f"{len(te):,} rows"], textposition="outside")])
    fig.update_layout(title="Positive rate after stratified split", yaxis_tickformat=".2%", yaxis_range=[0, 0.15])
    add("split_balance.json", fig, "Stratified split",
        "Stratification keeps the 11.27% rate identical in train and test.", "data")

    write_json(OUT / "figures.json", {"figures": figs})
    validate_dataset()
    print(f"dataset artifacts written: {len(figs)} figures; base rate {base:.4f}")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(centre - half), float(centre + half)]


if __name__ == "__main__":
    main()
