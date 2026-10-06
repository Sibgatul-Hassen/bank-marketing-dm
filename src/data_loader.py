"""Loading, integrity checking and splitting. OWNER: Agent-00. Everyone imports this."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .preprocessing import clean, feature_lists


class IntegrityError(RuntimeError):
    pass


def file_sha256(path=C.DATA_RAW) -> str:
    """SHA256 of the file with line endings normalised to LF."""
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def load_raw(verify: bool = True) -> pd.DataFrame:
    """Read the UCI CSV. Raises IntegrityError if the hash or shape is wrong."""
    raw = C.DATA_RAW.read_bytes().replace(b"\r\n", b"\n")
    if verify:
        digest = hashlib.sha256(raw).hexdigest()
        if digest != C.EXPECTED_SHA256:
            raise IntegrityError(
                f"SHA256 mismatch: got {digest}, expected {C.EXPECTED_SHA256}. "
                "You are not working on the same data as the rest of the team."
            )
    df = pd.read_csv(io.BytesIO(raw), sep=C.CSV_SEP)
    if verify and df.shape != C.EXPECTED_SHAPE:
        raise IntegrityError(f"Shape {df.shape} != expected {C.EXPECTED_SHAPE}")
    return df


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    categorical: list[str]
    numeric: list[str]
    encoding: str
    include_duration: bool
    # Only for encoding="embedding": integer codes fitted on train, plus TabNet's cat_idxs/cat_dims.
    vocab: dict[str, list[str]] = field(default_factory=dict)
    cat_idxs: list[int] = field(default_factory=list)
    cat_dims: list[int] = field(default_factory=list)
    test_index: np.ndarray | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)


def split_indices(n: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The single stratified 80/20 split every model uses."""
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    return tr, te


def load_processed(include_duration: bool = False, encoding: str = "onehot") -> Dataset:
    """Cleaned, split data.

    encoding:
      "onehot"    -> categoricals left as strings; fit `preprocessing.make_onehot_preprocessor`
                     inside your pipeline so the vocabulary and scaler are learned on train only.
      "native"    -> categoricals as strings for CatBoost's own encoder.
      "embedding" -> categoricals as integer codes (vocabulary from train only) for TabNet / TabPFN.
    """
    if encoding not in {"onehot", "native", "embedding"}:
        raise ValueError(f"unknown encoding {encoding!r}")
    df = clean(load_raw(), include_duration=include_duration)
    y = df.pop(C.TARGET).to_numpy()
    cats, nums = feature_lists(include_duration)
    X = df[cats + nums]
    tr, te = split_indices(len(X), y)
    X_train, X_test = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)

    ds = Dataset(X_train, X_test, y[tr], y[te], cats, nums, encoding, include_duration, test_index=te)
    if encoding == "embedding":
        X_train, X_test = X_train.copy(), X_test.copy()
        for c in cats:
            levels = sorted(X_train[c].unique())
            ds.vocab[c] = levels
            mapping = {v: i for i, v in enumerate(levels)}
            X_train[c] = X_train[c].map(mapping).astype(int)
            # A level unseen in train maps to 0; none exist with this split, but stay safe.
            X_test[c] = X_test[c].map(mapping).fillna(0).astype(int)
        ds.X_train, ds.X_test = X_train, X_test
        ds.cat_idxs = [X_train.columns.get_loc(c) for c in cats]
        ds.cat_dims = [len(ds.vocab[c]) for c in cats]
    return ds


if __name__ == "__main__":
    print("sha256 (LF-normalised):", file_sha256())
    df = load_raw()
    print("OK", df.shape)
