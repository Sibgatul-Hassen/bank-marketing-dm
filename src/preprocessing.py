"""The four data-problem fixes from §3.3. OWNER: Agent-00. Frozen after Wave 0.

`clean()` is purely row-wise — it learns nothing from the data — so it is safe to apply before
the train/test split. Everything that *is* fitted (one-hot vocabulary, scaling) lives in
`make_onehot_preprocessor()` and must be fitted inside a pipeline / CV fold.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def clean(df: pd.DataFrame, include_duration: bool = False) -> pd.DataFrame:
    df = df.copy()

    # Problem 1 — duration is known only after the call ends. Excluded unless explicitly asked for.
    if not include_duration:
        df = df.drop(columns=[C.LEAK_FEATURE])

    # Problem 2 — pdays == 999 means "never contacted", not "contacted 999 days ago".
    # The flag carries that information; the numeric column keeps real day counts (0–27) and
    # uses -1 as a placeholder for "not applicable", outside the valid range.
    contacted = df["pdays"] != C.PDAYS_SENTINEL
    df["was_contacted"] = contacted.astype(int)
    df["pdays"] = df["pdays"].where(contacted, -1)

    # Problem 3 — `unknown` is kept as its own level (it carries signal: education=unknown
    # subscribes at 14.50%). Nothing to do here; documented in results/dataset/quality.json.

    # Problem 4 — `default` has 3 "yes" rows out of 41,188. Keep only whether it is unknown.
    df["default_unknown"] = (df["default"] == "unknown").astype(int)
    df = df.drop(columns=["default"])

    df[C.TARGET] = (df[C.TARGET] == "yes").astype(int)
    for c in C.CATEGORICAL:
        df[c] = df[c].astype(str)
    return df


def feature_lists(include_duration: bool = False) -> tuple[list[str], list[str]]:
    nums = list(C.NUMERIC) + list(C.BINARY)
    if include_duration:
        nums = [C.LEAK_FEATURE] + nums
    return list(C.CATEGORICAL), nums


def make_onehot_preprocessor(categorical, numeric, scale: bool = True) -> ColumnTransformer:
    """One-hot + (optional) standard scaling. Fit inside the fold, never before the split."""
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical)),
            ("num", num, list(numeric)),
        ],
        verbose_feature_names_out=False,
    )


def onehot_source_map(feature_names_out, categorical) -> dict[str, str]:
    """Map each one-hot output column (e.g. 'job_student') back to its source feature ('job')."""
    out = {}
    for f in feature_names_out:
        src = next((c for c in categorical if f.startswith(c + "_")), f)
        out[f] = src
    return out
