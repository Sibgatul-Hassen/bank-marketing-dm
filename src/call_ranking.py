"""Who to call first — per model: what score it ranks clients by, and what calling its top-k would achieve.
Writes results/models/<id>/call_ranking.json.   python -m src.call_ranking

Reads each model's saved test-set scores (predictions.json); never retrains. All numbers are on the
8,238 held-out test clients, without duration."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .contract import write_json
from .data_loader import load_raw

SHARES = [0.05, 0.10, 0.20, 0.30]

METHOD = {
    "decision_tree": {
        "score": "the subscription rate of the leaf the client lands in",
        "how": "Each client is sent down the tree, answering one question per level, until they reach a leaf. Their score is the "
               "(class-balanced) share of subscribers among the training clients in that leaf. Every client in the same leaf gets the "
               "same score, so the call list is really a ranking of leaves — clients in the best leaf are called first.",
        "steps": ["Follow the client's path from the root, one question per level", "Stop at a leaf",
                  "Score = share of subscribers among training clients in that leaf", "Sort leaves best first; call down the list"],
    },
    "catboost": {
        "score": "the probability from summing hundreds of small trees",
        "how": "Every tree in the ensemble looks at the client and adds a small amount to their log-odds of subscribing; the total is "
               "turned into a probability with the sigmoid function. Because hundreds of trees each contribute, almost every client "
               "gets a different score, so the list is a fine-grained ranking of individual clients.",
        "steps": ["Start from the base-rate log-odds (−2.06)", "Each tree adds or subtracts a little, scaled by the learning rate",
                  "Convert the sum to a probability with the sigmoid", "Sort clients by probability; call down the list"],
    },
    "tabnet": {
        "score": "the probability from the network after four attention steps",
        "how": "The client's features pass through four decision steps; at each one the network attends to a few features (its mask) "
               "and adds to its decision. The final layer turns that into a probability of subscribing, and clients are sorted by it.",
        "steps": ["Embed categorical features, normalise numeric ones", "Four steps each pick a few features and add to the decision",
                  "Final layer → probability of subscribing", "Sort clients by probability; call down the list"],
    },
    "tabpfn": {
        "score": "the posterior probability given 3,000 example clients",
        "how": "The client is placed next to 3,000 training clients whose outcomes are known, and the pre-trained transformer reads the "
               "whole table at once to estimate how likely this client is to subscribe. No training happens; the score is a single "
               "forward pass. Its probabilities are well calibrated, so a score of 0.30 really means roughly a 30% chance.",
        "steps": ["Give the model 3,000 known clients as context", "Append the client to score",
                  "One forward pass → probability of subscribing", "Sort clients by probability; call down the list"],
    },
    "umap_hdbscan": {
        "score": "the subscription rate of the client's cluster",
        "how": "The clustering never sees who subscribed. To turn it into a call list we place each client in their HDBSCAN cluster and "
               "score them by how often training clients in that cluster subscribed. All clients in a cluster tie, so you call whole "
               "clusters at a time, best cluster first.",
        "steps": ["Find the client's cluster on the UMAP map", "Look up that cluster's subscription rate (training clients only)",
                  "Score = that rate; everyone in the cluster ties", "Call the best cluster first, then the next"],
    },
    "fp_growth": {
        "score": "the confidence of the best rule the client matches",
        "how": "Each actionable rule (→ y=yes, lift ≥ 1.2) is a condition like 'poutcome=success'. A client's score is the confidence of "
               "the strongest rule whose conditions they all meet; clients matching no rule get the base rate (11.3%). Most clients match "
               "no rule, so the list has a short, strong head and a long flat tail.",
        "steps": ["Check which actionable rules the client satisfies", "Score = highest confidence among those rules",
                  "No rule matched → base rate 11.3%", "Sort by score; call rule-matched clients first"],
    },
}


def within(df, col):
    aucs, w = [], []
    for _, g in df.groupby(col):
        if g.y.nunique() == 2 and len(g) > 50:
            aucs.append(roc_auc_score(g.y, g.s))
            w.append(len(g))
    return float(np.average(aucs, weights=w))


def main():
    raw = load_raw()
    for mid in C.MODEL_IDS:
        p = C.RESULTS_MODELS / mid / "predictions.json"
        if not p.exists():
            print(f"  skip {mid}: no predictions.json")
            continue
        pr = json.loads(p.read_text())
        y = np.array(pr["y_true"])
        s = np.array(pr["without_duration"]["proba"], dtype=float)
        idx = np.array(pr["test_index"])
        n, base = len(y), float(y.mean())
        order = np.lexsort((np.arange(n), -s))   # stable: ties keep test order
        rows = []
        for q in SHARES:
            k = int(round(q * n))
            top = y[order[:k]]
            rows.append({"share": q, "calls": k, "hit_rate": float(top.mean()), "lift": float(top.mean() / base),
                         "captured": float(top.sum() / y.sum())})
        df = pd.DataFrame({"y": y, "s": s, "month": raw.month.values[idx], "regime": raw["nr.employed"].values[idx]})
        auc = float(roc_auc_score(y, s))
        wm, wr = within(df, "month"), within(df, "regime")
        out = {"model_id": mid, **METHOD[mid], "test_clients": n, "subscribers": int(y.sum()), "base_rate": base,
               "roc_auc": auc, "distinct_scores": int(len(np.unique(s))), "top": rows,
               "within_month_auc": wm, "within_regime_auc": wr}
        pred = pr["without_duration"].get("pred")
        if pred is not None:
            out["threshold_call_share"] = float(np.mean(pred))
        r20 = rows[2]
        out["summary"] = (f"Calling the top 20% of clients by {out['score']} ({r20['calls']:,} calls) reaches "
                          f"{r20['captured']:.0%} of all subscribers, with {r20['hit_rate']:.0%} of calls ending in a subscription — "
                          f"{r20['lift']:.1f}× calling at random.")
        out["caveat"] = (f"Much of this ranking is about when a client was called: among clients called under the same economic "
                         f"conditions its AUC falls from {auc:.3f} to {wr:.3f} (within the same month: {wm:.3f}). A bank choosing whom "
                         "to call this week works within one period, so expect less separation than the table suggests. Offline estimate "
                         "on a random split of 2008–2010 data, not a live campaign.")
        write_json(C.RESULTS_MODELS / mid / "call_ranking.json", out)
        print(f"  {mid}: top-20% captures {r20['captured']:.1%}, within-regime AUC {wr:.3f}, {out['distinct_scores']} distinct scores")


if __name__ == "__main__":
    main()
