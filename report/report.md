# A Leakage-Aware, Visually Explained Comparison of Modern Tabular Learning Methods for Bank Telemarketing

*Data Mining · United International University · 2026 · © Sibgatul Hassen*

> Generated from `results/` by `python -m src.report`. Every figure below is read from the committed artifacts.

## Abstract

A Portuguese bank sells term deposits by telephone; only 11.27% of 41,188 contacted clients subscribe. We compare six techniques — a decision tree, CatBoost, TabNet, TabPFN, UMAP + HDBSCAN and FP-Growth — under one leakage-aware protocol. The call duration, the most predictive column, is known only after the call, so every classifier is trained with and without it. Without the leak the best ranking is CatBoost's (PR-AUC 0.4786, ROC-AUC 0.8109). Random Forest gains +0.166 AUC from the leak; Gaussian NB only +0.068 (2.4×). On PR-AUC: Random Forest +0.236 vs Gaussian NB +0.041 (5.8×). Random Forest falls from rank 2 to 5 when duration is removed; kNN (k=25) rises from 5 to 4. A comparison run on leaky data therefore measures each algorithm's capacity to exploit the leak rather than its predictive ability.

## 1. Data

UCI Bank Marketing (Moro, Cortez & Rita, 2014), `bank-additional-full.csv`: 41,188 × 21, May 2008 – November 2010. SHA256 `233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df` (verified on every load). Majority-class baseline accuracy 0.8873.

| Problem | Evidence | Action |
|---|---|---|
| duration is a target leak | r = 0.405 with target; min call 0 s for 'no' vs 37 s for 'yes' | Excluded from every model. Reintroduced only for the with-duration comparison run. |
| pdays = 999 is a sentinel, not a number | 39,673 rows (96.32%) have pdays = 999 | Split into binary was_contacted plus numeric pdays (999 replaced by -1, a placeholder outside the real 0–27 range; the flag carries the meaning). |
| 'unknown' is missing data disguised as a category | 12,718 'unknown' cells in 6 columns; metadata says no missing values | Kept as its own level (one of three defensible choices: keep / impute / drop). education=unknown subscribes above several known levels, so it carries signal. |
| default is effectively constant | counts no 32,588, unknown 8,597, yes 3 | Dropped the three-level column; kept a binary default_unknown flag. |

**Split protocol.** Stratified 80/20 holdout (random_state=42). Stratified 5-fold CV on the training part only. All scaling, encoding vocabularies and class weighting are fitted inside each fold — never before the split.

## 2. Methods

- **Decision Tree** (Breiman et al., Classification and Regression Trees (1984); Quinlan, ID3 (1986), C4.5 (1993)). Plays twenty questions: asks the most informative question first, then repeats on each group.
- **CatBoost** (Prokhorenkova et al., CatBoost: unbiased boosting with categorical features, NeurIPS 2018). Builds trees in sequence, each correcting the previous one's mistakes.
- **TabNet** (Arık & Pfister, TabNet: Attentive Interpretable Tabular Learning, AAAI 2021). A neural network that, at each step, chooses a handful of features to look at — and lets you read which ones.
- **TabPFN** (Hollmann et al., Accurate predictions on small data with a tabular foundation model, Nature 637 (2025); TabPFN-2.5 (Prior Labs, 2025)). A transformer pre-trained on millions of synthetic tables; it predicts in one forward pass, with no training and no tuning.
- **UMAP + HDBSCAN** (McInnes, Healy & Melville, UMAP (2018); Campello, Moulavi & Sander, HDBSCAN (2013); McInnes & Healy, hdbscan (2017)). Flattens 50 dimensions into 2–3 while keeping neighbours together, then finds dense groups without being told how many.
- **FP-Growth** (Han, Pei & Yin, Mining frequent patterns without candidate generation, SIGMOD 2000; Agrawal & Srikant, Apriori, VLDB 1994). Finds attribute combinations that occur often, then keeps the rules that beat chance — ranked by lift, not confidence.

SHAP (exact TreeSHAP for the tree and CatBoost; Kernel SHAP for TabNet and TabPFN, with model calls batched — validated against exact TreeSHAP at per-value r = 0.97–0.998) is computed offline and summed back to the 20 cleaned features (exact by additivity), so all models are explained in the same vocabulary.

## 3. Results

Without duration (primary). Thresholds maximise F1 on training-side predictions; the test set is never used for any choice.

| Model | Acc | Prec | Rec | F1 | ROC-AUC | PR-AUC | MCC | Brier | Train + predict s | Leak Δ AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| *Majority baseline* | 0.8874 | — | 0 | 0 | 0.5000 | 0.1126 | 0 | — | — | — |
| CatBoost | 0.8846 | 0.4893 | 0.5668 | 0.5252 | 0.8109 | 0.4786 | 0.4615 | 0.1542 | 21.0 | +0.1450 |
| TabPFN | 0.8734 | 0.4523 | 0.5873 | 0.5110 | 0.8067 | 0.4611 | 0.4447 | 0.0769 | 637.6 | +0.1421 |
| TabNet | 0.8735 | 0.4520 | 0.5787 | 0.5076 | 0.8017 | 0.4458 | 0.4406 | 0.1583 | 251.4 | +0.1478 |
| Decision Tree | 0.8727 | 0.4510 | 0.6002 | 0.5150 | 0.7954 | 0.4431 | 0.4494 | 0.1590 | 0.3 | +0.1510 |
| UMAP + HDBSCAN *(cluster-rate AUC (test))* | | | | | 0.7930 | | | | | |
| FP-Growth *(rule-score AUC (test))* | | | | | 0.7590 | | | | | |

### Who to call first

Each model ranks the held-out clients by its own score; the bank calls down the list.

| Model | Ranks by | Top 10%: subscribe | Top 20%: subscribe | Top 20%: subscribers reached | AUC within same economic period |
|---|---|---|---|---|---|
| Decision Tree | the subscription rate of the leaf the client lands in | 50.5% | 36.4% | 64.7% | 0.595 (overall 0.795) |
| CatBoost | the probability from summing hundreds of small trees | 52.8% | 37.6% | 66.7% | 0.606 (overall 0.811) |
| TabNet | the probability from the network after four attention steps | 48.9% | 36.3% | 64.4% | 0.592 (overall 0.802) |
| TabPFN | the posterior probability given 3,000 example clients | 52.3% | 36.3% | 64.4% | 0.616 (overall 0.807) |
| UMAP + HDBSCAN | the subscription rate of the client's cluster | 49.5% | 34.7% | 61.6% | 0.565 (overall 0.793) |
| FP-Growth | the confidence of the best rule the client matches | 48.1% | 34.6% | 61.4% | 0.551 (overall 0.759) |

Random calling converts at 11.3%. Much of every ranking reflects when a client was called: within one economic period the separation drops sharply, so a bank choosing whom to call this week should expect less than the overall figures suggest.

## 4. The leakage finding

Six classical families (the plan's §3.6), re-measured in our pipeline, ranked by ROC-AUC:

| Family | AUC with | AUC without | Δ AUC | Δ PR-AUC | Rank with → without |
|---|---|---|---|---|---|
| HistGradientBoosting | 0.9539 | 0.8105 | +0.1434 | +0.2037 | 1 → 1 |
| Random Forest | 0.9462 | 0.7800 | +0.1661 | +0.2362 | 2 → 5 |
| Logistic Regression | 0.9425 | 0.8009 | +0.1416 | +0.1705 | 3 → 2 |
| Decision Tree (d=5) | 0.9380 | 0.7902 | +0.1478 | +0.1938 | 4 → 3 |
| kNN (k=25) | 0.9289 | 0.7822 | +0.1467 | +0.1648 | 5 → 4 |
| Gaussian NB | 0.8432 | 0.7752 | +0.0680 | +0.0410 | 6 → 6 |

Random Forest gains +0.166 AUC from the leak; Gaussian NB only +0.068 (2.4×). On PR-AUC: Random Forest +0.236 vs Gaussian NB +0.041 (5.8×). Random Forest falls from rank 2 to 5 when duration is removed; kNN (k=25) rises from 5 to 4.

*Positioning.* Differential leakage sensitivity is established for duplicate leakage. We measure it for feature leakage on this benchmark and show it reorders the ranking.

## 5. Why each model ranked where it did

**CatBoost — rank 1 of 4 (PR-AUC 0.4786).** Boosting adds hundreds of shallow trees, each fitted to what the previous ones got wrong, so it can combine many weak signals (contact channel, month, the macro regime, poutcome) that a single tree must choose between. Ordered target statistics let one split separate high- from low-responding jobs or months without one-hot fragmentation. Its margin over the single tree is +0.0355 PR-AUC: real (see the McNemar test in the comparator) but small, because without duration the signal is concentrated in a handful of features that a depth-7 tree already reaches. Its leak inflation (+0.145 AUC) is typical of tree ensembles: duration is a single monotone feature that trees threshold precisely.

**Decision Tree — rank 4 of 4 (PR-AUC 0.4431).** One tree of axis-parallel questions. It ranks below the ensemble because each client follows a single path, so evidence from features off that path is ignored, and because the three collinear macro features cannot be used jointly. It stays close because the strongest legitimate signals — the economic period (nr.employed / euribor3m), poutcome=success and the contact channel — are exactly the kind of sharp thresholds a tree finds first. It is the only model whose every prediction can be traced by hand, which we do on its page.

**TabNet — rank 3 of 4 (PR-AUC 0.4458).** A neural network with sequential attention. It lands below CatBoost, as the plan predicted. With ~28,000 training rows, 20 features and a weak signal, a network has more parameters to fit than the data can pin down, and its smooth decision surface has no advantage over trees when the useful structure is a few sharp thresholds. It took 12.0× CatBoost's training time on CPU. Its value here is the masks: an attribution produced by the architecture itself, which we compare with SHAP on its page.

**TabPFN — rank 2 of 4 (PR-AUC 0.4611).** No training, no tuning, and only a 3,000-row context (one eleventh of the training data, a CPU limit) — yet ROC-AUC 0.8067, in the same band as models tuned on all 32,950 rows. A prior learned from millions of synthetic tables substitutes for data: on its context curve it beats CatBoost trained on the same rows at 6 of 6 sizes (e.g. 0.775 vs 0.764 at 250 rows). It is also the only well-calibrated model (Brier 0.0769), because it was not class-weighted. The cost is inference: every prediction re-reads the whole context, which is why the full-data run was not feasible without a GPU.

**UMAP + HDBSCAN — not ranked (unsupervised).** It never sees the label, so it cannot compete; scored by its clusters' subscription rates it reaches ROC-AUC 0.793. Its contribution is diagnostic: the clusters align with the month and economic regime of the call far more than with subscription — consistent with every supervised model leaning on the macro features: the dominant structure in this data is the calendar of the campaign.

**FP-Growth — not ranked (pattern mining).** Rules describe groups, not individuals: a client matching no actionable rule gets the base-rate score, so as a ranker it reaches only ROC-AUC 0.759. But its top rules (poutcome=success, cellular contact in low-volume months, students and retirees) are readable policies with measured lift and coverage. They overlap with what SHAP ranks highly — the low-rate economic period, contact channel and month — though SHAP puts poutcome lower than the rules do, because few clients had a previous success.

**The reference families — why the ranking reorders.** The six reference families explain the leakage finding. Random Forest gains the most from duration (+0.166 AUC) and falls from rank 2 to 5 without it: fully grown trees carve duration into fine thresholds that nearly determine the label, and without it the same unpruned trees memorise noise in a weak signal. Gaussian NB gains least (+0.068): it multiplies independent per-feature likelihoods, so duration is one vote among twenty, and its Gaussian model of a heavily skewed duration distribution captures little of it. kNN gains +0.147; the plan's §3.6 reported a smaller with-duration AUC (0.8815) than we reproduce (0.9289, scaled or unscaled) — an unresolved discrepancy we report rather than hide. The general lesson: a leaky feature rewards the models best able to exploit one sharp signal, so leaky benchmarks rank exploitation, not prediction.

## 6. What the models rely on (SHAP)

- **Decision Tree** (TreeExplainer, 5,000 clients): nr.employed (42%), cons.conf.idx (15%), cons.price.idx (13%), euribor3m (6%), pdays (5%). Macro features: 78% of attribution.
- **CatBoost** (TreeExplainer (CatBoost native TreeSHAP), 5,000 clients): nr.employed (19%), contact (14%), month (12%), euribor3m (11%), emp.var.rate (7%). Macro features: 44% of attribution.
- **TabNet** (Kernel SHAP (batched), 500 clients): nr.employed (25%), month (16%), euribor3m (13%), cons.conf.idx (7%), contact (6%). Macro features: 50% of attribution.
  - TabNet masks vs SHAP: Spearman ρ = 0.38; shared top-5: euribor3m, month, nr.employed.
- **TabPFN** (Kernel SHAP (batched), 100 clients): nr.employed (17%), cons.conf.idx (17%), euribor3m (14%), month (10%), emp.var.rate (8%). Macro features: 64% of attribution.

SHAP shows what a model uses, not what causes subscription.

## 7. Findings per page, including failures

### Decision Tree

A depth-7 tree reaches AUC 0.795 without the leaky call length — and 0.946 with it.

- *result* — Without duration: ROC-AUC 0.7954, PR-AUC 0.4431, F1 0.5150 (threshold 0.669 chosen on out-of-fold predictions). The plan's d=5 smoke test measured 0.7902.
- *result* — Leak inflation: +0.1510 AUC and +0.2288 PR-AUC when duration is allowed. The CV-best depth is 7 in both runs: the leak changes which splits are made, not how deep the tree should be.
- *result* — Hand-computed entropy, information gain, split information and Gini for poutcome match scipy and sklearn to 1e-9. Information gain picks poutcome; C4.5's gain ratio (after the average-gain filter) picks poutcome.
- *result* — The depth curve is flat: depth 3 CV AUC 0.7699 vs best 0.7820 at depth 7. The plan's prediction (shallow ≈ deep) holds — the signal without duration is weak and diffuse.
- *result* — day_of_week ranks 7 of 20 in SHAP attribution — the plan predicted near-zero importance; not confirmed for this model: a tree uses it for a few late splits even though its overall rates are nearly flat.
- *limitation* — Axis-parallel splits cannot use the three collinear macro features jointly; the tree picks one (usually nr.employed or euribor3m) and the others become nearly redundant.
- *limitation* — Balanced class weights push accuracy at threshold 0.5 below the 0.8873 majority baseline; at the F1-optimal threshold accuracy is 0.8727. Probabilities are not calibrated (Brier 0.1590).
- *limitation* — A random 80/20 split ignores that the data is time-ordered (May 2008 – Nov 2010); a temporal split would likely score lower.

### CatBoost

CatBoost reaches AUC 0.811 without the leak — the strongest legitimate ranking so far, but by a modest margin.

- *result* — Without duration: ROC-AUC 0.8109, PR-AUC 0.4786, F1 0.5252; 5-fold CV AUC 0.7968 ± 0.0048. The plan predicted 0.80–0.82 (HistGradientBoosting measured 0.8136).
- *result* — Leak inflation: +0.1450 AUC, +0.2247 PR-AUC. With duration early stopping ran to 335 trees instead of 252.
- *result* — Against the decision tree: PR-AUC 0.4786 vs 0.4431 (margin +0.0355). McNemar on test predictions: p = 8.53e-10 — the difference in errors is statistically significant at 0.05.
- *limitation* — Balanced weights leave probabilities over-confident (mean prediction 0.38 vs base rate 0.11, Brier 0.1542). Use the scores for ranking, or recalibrate before reading them as chances.
- *limitation* — Ordered target statistics depend on a random permutation; rare levels (education=illiterate, 18 rows) get noisy encodings dominated by the prior.
- *limitation* — Gains over a single tree come with ~100× more trees and much less transparency; the extracted tree is one of hundreds.

### TabNet

TabNet reaches AUC 0.802 — a neural network that shows its own feature choices, at more cost and no gain over trees.

- *result* — Without duration: ROC-AUC 0.8017, PR-AUC 0.4458, F1 0.5076; trained in 251s on CPU.
- *result* — Leak inflation: +0.1478 AUC, +0.2327 PR-AUC.
- *result* — Masks vs SHAP: Spearman ρ = 0.38; shared top-5 features: euribor3m, month, nr.employed. The two methods disagree substantially — a genuine result: masks show what the network looks at, SHAP what changes its output.
- *failure* — TabNet loses to CatBoost: PR-AUC 0.4458 vs 0.4786, ROC-AUC 0.8017 vs 0.8109, at 12.0× the training time. We did not tune TabNet further to change this.
- *limitation* — No cross-validation (CPU cost); the single-holdout AUC has no error bar, and neural nets vary more across seeds than trees.
- *limitation* — SHAP for TabNet uses Kernel SHAP (batched; same estimator as shap.KernelExplainer) on 500 clients with 200 coalition samples (1.0 min) — approximate, unlike exact TreeSHAP.
- *limitation* — Masks show where the network looks, not how a feature moves the prediction; a heavily attended feature can still have a small effect.

### TabPFN

With no tuning and only 3,000 rows, TabPFN scores AUC 0.807 — in the same band as the tuned models trained on all 32,950.

- *result* — Without duration: ROC-AUC 0.8067, PR-AUC 0.4611, F1 0.5110, Brier 0.0769 (well calibrated — no class weighting).
- *result* — Leak inflation: +0.1421 AUC, +0.2204 PR-AUC.
- *result* — Data efficiency: at 250 rows TabPFN 0.775 vs CatBoost 0.764 vs tree 0.625. Best TabPFN point on the curve: 0.811 at 3,000 rows.
- *failure* — The plan's model, TabPFN-2.5, could not be run: its weights are licence-gated. We used TabPFN v2 instead and say so on every page.
- *failure* — The full-context run (41k rows) and the 10k/20k/41k points of the context curve were not feasible without a GPU; the curve stops at 5,000.
- *limitation* — Prediction is slow: 636 s for 8,238 clients on CPU, versus well under a second for the tree. SHAP needed 46 minutes for 100 clients, even batched; the library's per-client KernelExplainer ran for over three hours without finishing and was abandoned.
- *limitation* — No cross-validation: CV would mean 5 more full inference passes. The single holdout AUC has no error bar here.

### UMAP + HDBSCAN

The two classes do not separate. Clients cluster by the economic period of the call — the finding the plan predicted.

- *result* — 26 clusters, 0.3% noise. Adjusted mutual information with the clusters: month 0.631, macro regime (5 macro features) 0.551, poutcome 0.268, contact 0.243, y (subscription) 0.042, job 0.038.
- *result* — Best cluster subscribes at 74.1% (765 clients): was_contacted high (mean 1 vs 0.03678); previous high (mean 2 vs 0.173); poutcome=success (89% vs 3% overall). Subscription 74.1% — high response.
- *result* — Using only 'which cluster are you in?' ranks held-out clients at AUC 0.793 — real but weaker than the supervised models (~0.80).
- *failure* — No region of the map is predominantly subscribers. The plan predicted this; we did not re-tune n_neighbors/min_dist to make one appear.
- *limitation* — The five macro features take only 375 distinct value combinations across 41,188 clients — they are period labels in disguise, and after scaling they act like a strong categorical signal that dominates Euclidean distance.
- *limitation* — Euclidean distance on one-hot columns treats every categorical mismatch as equally far. Gower distance or CatBoost embeddings might give different structure.
- *limitation* — UMAP distances between islands are not meaningful; only neighbourhoods are. Do not read the gap between two clusters as a measure of difference.

### FP-Growth

Rules that beat chance exist — poutcome=success lifts subscription 5.8× — but the highest-confidence rules are worthless.

- *result* — 90,183 frequent itemsets (≤ 4 items) and 367,138 rules at confidence ≥ 0.3; 121 predict y=yes, 121 of them with lift ≥ 1.2.
- *result* — Of 121 rules with lift ≥ 1.2, only 20 survive the minimum-improvement filter: most long rules are a strong short rule (usually poutcome=success) plus an item that adds nothing. Ranking by lift alone fills the top of the list with such restatements.
- *result* — Plan's expected rules, verified on training data: poutcome=success → y=yes lift 5.76 (coverage 3.4%); job=student → y=yes lift 2.71 (coverage 2.2%); job=retired → y=yes lift 2.26 (coverage 4.1%); contact=cellular + month=mar → y=yes lift 4.58 (coverage 1.2%); contact=cellular + month=sep → y=yes lift 4.34 (coverage 1.2%); contact=cellular + month=oct → y=yes lift 3.95 (coverage 1.4%); contact=cellular + month=dec → y=yes lift 4.66 (coverage 0.4%).
- *result* — Rules generalise: the top rule's confidence is 0.65 on training clients and 0.66 on 268 held-out clients.
- *result* — Contrary to the textbook expectation, mlxtend's low-memory Apriori was 2.0× faster than its FP-Growth at 1% support. Apriori/FP-Growth time ratio by support — 10%: 0.81×, 5%: 1.18×, 3%: 0.91×, 2%: 0.67×, 1%: 0.49×. With only 77 items and itemsets capped at 4, candidate generation stays cheap; FP-Growth's advantage grows with longer patterns.
- *failure* — mlxtend's default Apriori ran out of memory at 1% support: its level-4 step tried to allocate a 32,950 × 96,363 boolean matrix (2.96 GiB). We re-ran every support level with low_memory=True — candidate explosion, observed directly.
- *result* — Implementation matters as much as algorithm: our bit-packed Apriori (rows packed into 64-bit words, support = popcount) took 1.4s at 1% — faster than both library implementations. FP-Growth's advantage is over naive candidate counting, not over every Apriori.
- *limitation* — The confidence trap: loan=no → y=no has confidence 0.887 and lift 1.000. It restates the base rate.
- *limitation* — As a classifier the rules are weak (AUC 0.759): most clients match no actionable rule, so they all get the same score.
- *limitation* — Rules are only as good as the bins: the macro terciles are uneven because those features take few distinct values.

## 8. Deviations from the plan, stated openly

- TabPFN-2.5 weights are licence-gated; without a Prior Labs token we used TabPFN v2 (Hollmann et al., Nature 2025). On CPU the context was a stratified 3,000-row subsample and the context curve stops at 5,000 rows.
- The plan's kNN result with duration (0.8815) could not be reproduced: we measure 0.929 scaled and 0.926 unscaled. Reported, not resolved.
- The plan names `previous` (0.230) as duration's runner-up correlate; by absolute value it is nr.employed (−0.355) and pdays (−0.325).
- The plan predicted near-zero importance for day_of_week; the decision tree ranks it 7th of 20 in SHAP, so the prediction is not confirmed for every model.
- mlxtend's default Apriori ran out of memory at 1% support; low_memory mode was used, and was faster than mlxtend's FP-Growth there — contrary to the textbook expectation.
- TabNet and TabPFN have no cross-validation (CPU cost); their AUCs are single-holdout numbers.
- Windows Smart App Control blocked the newest numba and scikit-learn DLLs on the build machine; versions are pinned to ones that load (numba 0.62.1, scikit-learn 1.8.0).

## 9. Limitations

One bank, May 2008 – November 2010, across the financial crisis; five macro features encode that period, so nothing here transfers to another bank or decade. No novelty is claimed for the dataset. Deployment is simulated: the lift curve is an offline estimate. The `unknown` handling is one of three defensible choices. A random split ignores the time ordering of the calls; a temporal split would likely score lower.

## References

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.  
Breiman et al., Classification and Regression Trees (1984); Quinlan, ID3 (1986), C4.5 (1993).  
Prokhorenkova et al., CatBoost: unbiased boosting with categorical features, NeurIPS 2018.  
Arık & Pfister, TabNet: Attentive Interpretable Tabular Learning, AAAI 2021.  
Hollmann et al., Accurate predictions on small data with a tabular foundation model, Nature 637 (2025); TabPFN-2.5 (Prior Labs, 2025).  
McInnes, Healy & Melville, UMAP (2018); Campello, Moulavi & Sander, HDBSCAN (2013); McInnes & Healy, hdbscan (2017).  
Han, Pei & Yin, Mining frequent patterns without candidate generation, SIGMOD 2000; Agrawal & Srikant, Apriori, VLDB 1994.  
