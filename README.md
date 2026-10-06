# Bank Marketing — a leakage-aware, visually explained comparison of tabular learning methods

Data Mining course project (Track 1 + Track 2 delivery). UCI Bank Marketing, `bank-additional-full.csv`
(41,188 × 21, Moro, Cortez & Rita 2014).

**Claim.** The accuracy usually reported on this benchmark depends on `duration` — the call length, known
only after the call. We train every model with and without it, show the inflation is unequal across model
families, and show it **reorders the ranking**.

## Layout

```
src/                 Python: training offline, writes JSON artifacts
  config.py data_loader.py preprocessing.py   shared pipeline (frozen)
  contract.py        artifact writers + validators (§6)
  evaluation.py      the one metrics function every page uses
  explain.py         SHAP engine (aggregated to the 20 cleaned features)
  dataset_report.py  Data & EDA artifacts
  models/            one script per model page (+ reference_models.py for the §3.6 leakage table)
  comparison.py      leaderboard, leakage, McNemar, radar
  run_all.py         rebuild everything
results/             COMMITTED — the evidence; the frontend renders only this
frontend/            Vite + React + TS: framer-motion, Plotly, React Three Fiber, KaTeX, zustand
data/raw/            the CSV (SHA256 below)
```

## Run

```bash
python -m venv .venv && .venv/Scripts/activate        # Python 3.13
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.data_loader                 # verifies SHA256 233e260d…bd30df
python -m src.run_all --skip-slow         # everything except TabNet/TabPFN (~15 min)
python -m src.run_all                     # full rebuild (~3–4 h on 4 CPU cores)
python -m src.contract validate all       # artifact contract — run before every PR

cd frontend && npm install && npm run dev # http://localhost:5173
npm run check:katex                       # every formula renders
npm run build                             # static site in frontend/dist (results copied in)
```

**TabPFN-2.5** weights are licence-gated: accept the licence at ux.priorlabs.ai, then
`export TABPFN_TOKEN=...` and rerun `python -m src.run_all tabpfn`. Without a token the ungated
**TabPFN v2** weights are used, and every page says so.

## Integrity

`SHA256 233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df` — `.gitattributes` keeps the CSV
byte-identical across OSes; the loader also normalises CRLF before hashing.

## Protocol

Stratified 80/20 holdout (`random_state=42`), stratified 5-fold CV inside the training 80 %. Row-wise cleaning
(drop `duration`, split the `pdays=999` sentinel, collapse `default`) learns nothing from data and runs before the
split; every fitted step (one-hot vocabulary, scaling, class weighting, thresholds) is fitted on training data only.
Decision thresholds maximise F1 on out-of-fold/validation predictions, never on the test set.

© Sibgatul Hassen · Data Mining · United International University · 2026
