# Bank Marketing: Leakage-Aware Tabular Learning

Data Mining project by **Sibgatul Hassen** at United International University (2026).

This project studies whether machine-learning models can identify bank customers who are likely to subscribe to a term deposit after a telephone marketing campaign. It combines an offline Python experiment pipeline with a React/Vite frontend that presents the results as an interactive report.

## Project question

The UCI Bank Marketing dataset contains a feature called `duration`: the length of the current phone call. It is highly predictive of the target, but it is only known **after** the call has finished. Using it to decide whom to call is therefore target leakage.

The project evaluates every supervised model in two modes:

- **Without duration:** the realistic, pre-call prediction setting.
- **With duration:** a deliberately leaky comparison that measures how differently model families exploit the same leaked signal.

The main conclusion is that a benchmark containing `duration` can produce misleading rankings. Models benefit by different amounts, so the leak measures a model's ability to exploit the leak rather than its ability to select customers before calling them.

## Main findings

The primary evaluation uses the no-duration setting and a stratified 80/20 holdout:

| Model | ROC-AUC | PR-AUC | F1 | Notes |
|---|---:|---:|---:|---|
| CatBoost | 0.8109 | 0.4786 | 0.5252 | Best supervised model in the primary comparison |
| TabPFN | 0.8067 | 0.4611 | 0.5110 | Strong ranking and calibration, but very slow on CPU |
| TabNet | 0.8017 | 0.4458 | 0.5076 | Attention masks provide an additional explanation |
| Decision Tree | 0.7954 | 0.4431 | 0.5150 | Fastest and easiest model to trace manually |
| UMAP + HDBSCAN | 0.7930 | — | — | Unsupervised cluster-rate ranking |
| FP-Growth | 0.7590 | — | — | Rule-score ranking, not an individual classifier |

The target is imbalanced: only **11.27%** of the 41,188 contacted clients subscribed. Precision-recall AUC is therefore reported alongside ROC-AUC. Accuracy alone is not an adequate metric; the majority-class accuracy is approximately 0.8873.

The reference leakage experiment shows:

- Random Forest gains **+0.1661 ROC-AUC** from `duration`.
- Gaussian Naive Bayes gains only **+0.0680 ROC-AUC**.
- Random Forest moves from rank 2 to rank 5 when `duration` is removed.
- On PR-AUC, the leakage gain is **+0.236** for Random Forest versus **+0.041** for Gaussian Naive Bayes.

## Dataset

The project uses the UCI Bank Marketing `bank-additional-full.csv` dataset:

- 41,188 rows
- 21 original columns
- May 2008 to November 2010
- Target: whether the customer subscribed to a term deposit
- Source: Moro, Cortez & Rita (2014)

The loader verifies this SHA256 hash on every load:

```text
233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df
```

Important preprocessing decisions:

- `duration` is excluded from the primary no-leak model inputs.
- `pdays=999` is treated as a previous-contact sentinel. It becomes a `was_contacted` flag plus a numeric `pdays` value with the sentinel replaced.
- `unknown` values are retained as a meaningful category instead of being silently imputed or discarded.
- The nearly constant `default` field is removed and represented by a `default_unknown` indicator.
- One-hot vocabularies, scaling, class weighting, thresholds, and other fitted transformations are learned from training data only.

## Evaluation protocol

1. Load and hash-check the raw dataset.
2. Apply deterministic row-wise cleaning.
3. Create a stratified 80/20 train/test holdout with `random_state=42`.
4. Use stratified 5-fold cross-validation inside the training portion where the model supports it.
5. Fit preprocessing and model-selection steps on training data only.
6. Select F1 thresholds using training-side out-of-fold or validation predictions.
7. Evaluate once on the held-out test set.
8. Write validated JSON artifacts to `results/`.

Reported metrics include accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Matthews correlation coefficient, Brier score, training/prediction time, ranking-at-budget results, and leakage deltas where applicable.

## Models and analyses

### Supervised models

- **Decision Tree:** interpretable axis-aligned splits; includes depth curves and hand-computed entropy, information gain, gain ratio, and Gini checks.
- **CatBoost:** categorical-aware gradient boosting; the strongest legitimate supervised model in the current results.
- **TabNet:** attentive neural network with feature masks; mask importance is compared with SHAP importance.
- **TabPFN:** pretrained tabular foundation model; uses a bounded context on CPU and caches expensive predictions.

### Unsupervised and pattern-mining analyses

- **UMAP + HDBSCAN:** reduces the cleaned feature space and discovers density-based clusters without using the label. Cluster subscription rates are used only as a diagnostic ranking.
- **FP-Growth:** mines frequent attribute combinations and reports readable rules with support, confidence, lift, and coverage.

### Explanations and comparisons

- TreeSHAP is used for the decision tree and CatBoost.
- Batched Kernel SHAP is used for TabNet and TabPFN.
- Explanations are aggregated back to the 20 cleaned features so models use a common vocabulary.
- The comparison pipeline produces leaderboard, leakage, McNemar, radar, and pairwise artifacts.
- A call-ranking analysis reports subscription rates and subscriber reach at calling budgets such as the top 10% and top 20%.

SHAP describes which features influence a model's predictions. It does not establish that those features cause customers to subscribe.

## Repository layout

```text
data/
  raw/                         Original UCI CSV

src/
  config.py                    Paths and experiment constants
  data_loader.py               Dataset loading, cleaning, and hash validation
  preprocessing.py             Shared feature preparation
  evaluation.py                Shared metrics and threshold evaluation
  explain.py                   SHAP and explanation payload generation
  contract.py                  Artifact writers and validators
  dataset_report.py            Dataset and EDA artifact generation
  comparison.py                Leaderboard and cross-model comparisons
  call_ranking.py              Calling-budget analysis
  report.py                    Builds report/report.md from results/
  run_all.py                   Runs the complete offline pipeline
  models/                      One implementation per model/analysis

results/
  dataset/                     Dataset and EDA artifacts
  models/                      Per-model metrics, predictions, figures, and explanations
  comparison/                  Leaderboard, leakage, pairwise, and radar artifacts
  manifest.json                Artifact manifest

report/
  report.md                    Human-readable generated report

frontend/
  src/                         React application
  vite.config.ts               Serves results/ in development and copies it on build
  dist/                        Production build output (generated)
```

The frontend is intentionally read-only with respect to `results/`. Python writes the evidence; React renders it.

## Python setup

Python **3.13** is the tested version. From Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The dependency versions are pinned because they are the versions used to produce the committed artifacts. The project currently uses CPU inference by default.

Verify the dataset and artifacts:

```bash
python -m src.data_loader
python -m src.contract validate all
```

## Running the Python pipeline

Run the faster pipeline, excluding the slow TabNet and TabPFN stages:

```bash
python -m src.run_all --skip-slow
```

Run the complete pipeline:

```bash
python -m src.run_all
```

The full run can take several hours on a four-core CPU, especially for TabPFN SHAP and prediction. Expensive TabPFN stages are cached under `data/processed/tabpfn_cache`.

Run an individual model or analysis when needed:

```bash
python -m src.models.decision_tree
python -m src.models.catboost_model
python -m src.models.tabnet_model
python -m src.models.tabpfn_model
python -m src.models.umap_hdbscan
python -m src.models.fpgrowth_model
python -m src.comparison
python -m src.report
```

### TabPFN licensing and runtime

TabPFN-2.5 weights require a Prior Labs account and accepted license. Set the token before running the TabPFN stage:

```bash
export TABPFN_TOKEN="your-token"
python -m src.models.tabpfn_model
```

Without `TABPFN_TOKEN`, the pipeline uses the ungated TabPFN v2 weights. The generated page records which version was used. On CPU, the implementation intentionally uses a stratified context subset rather than the entire training set because full-context inference is impractical.

Never commit the token to Git or put it in the README.

## Running the frontend

The frontend requires Node.js 22 or a compatible current Node.js release:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

Useful frontend commands:

```bash
npm run check:katex
npm run lint
npm run build
npm run preview
```

`npm run build` creates `frontend/dist/` and copies the committed `results/` artifacts into `frontend/dist/results/`.

## GitHub Pages deployment

GitHub Pages works for the **frontend report only**. It does not execute Python, train models, run TabPFN, or provide an API.

The workflow in `.github/workflows/pages.yml`:

1. Checks out the repository.
2. Installs Node.js dependencies.
3. Builds the Vite frontend.
4. Uploads `frontend/dist` as a Pages artifact.
5. Deploys it to GitHub Pages.

The workflow deploys on pushes to `main` and can also be started manually from the Actions tab. In repository settings, select:

```text
Settings → Pages → Build and deployment → Source: GitHub Actions
```

The expected project-site URL is:

```text
https://sibgatul-hassen.github.io/bank-marketing-dm/
```

The frontend uses hash routing, so report pages remain refresh-safe on a static host.

## CI and validation

The existing GitHub Actions workflow validates:

- dataset hash
- artifact contract
- KaTeX formulas
- TypeScript compilation
- Vite production build

Before pushing a result update, run:

```bash
python -m src.data_loader
python -m src.contract validate all
cd frontend
npm ci
npm run check:katex
npm run build
```

## Reproducibility and limitations

- The primary split is random rather than temporal, although the data spans multiple economic periods.
- Duration is intentionally excluded from the realistic primary comparison.
- Class weighting improves minority detection but can make probabilities over-confident; ranking metrics should be interpreted separately from calibration.
- TabPFN uses a bounded CPU context and therefore is not a full-data comparison.
- UMAP + HDBSCAN and FP-Growth are diagnostic analyses, not directly comparable supervised classifiers.
- SHAP shows model attribution, not causality.
- The reported artifacts are generated offline and committed so the frontend can be deployed without a Python backend.

## License and attribution

Dataset: UCI Bank Marketing, Moro, Cortez & Rita (2014).

Model and method references are cited in [report/report.md](report/report.md). Project copyright © Sibgatul Hassen, 2026.
