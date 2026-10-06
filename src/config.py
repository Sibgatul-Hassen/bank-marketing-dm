"""Project-wide paths and constants. Shared infrastructure — change only by group decision."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "bank-additional-full.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS_DATASET = RESULTS / "dataset"
RESULTS_MODELS = RESULTS / "models"
RESULTS_COMPARISON = RESULTS / "comparison"

# SHA256 of the canonical UCI file (LF line endings). The loader normalises CRLF -> LF
# before hashing, because git's core.autocrlf rewrites line endings on Windows checkouts.
EXPECTED_SHA256 = "233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df"
EXPECTED_SHAPE = (41188, 21)
CSV_SEP = ";"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "y"
LEAK_FEATURE = "duration"
PDAYS_SENTINEL = 999

CATEGORICAL = [
    "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NUMERIC = ["age", "campaign", "pdays", "previous", *MACRO]
BINARY = ["was_contacted", "default_unknown"]

MODEL_IDS = ["decision_tree", "catboost", "tabnet", "tabpfn", "umap_hdbscan", "fp_growth"]
SUPERVISED_IDS = ["decision_tree", "catboost", "tabnet", "tabpfn"]

# Measured in the master plan (§3.2, §3.7) — used as regression tests.
BASE_RATE = 0.1127
MAJORITY_ACCURACY = 0.8873
AUC_RED_FLAG = 0.85
