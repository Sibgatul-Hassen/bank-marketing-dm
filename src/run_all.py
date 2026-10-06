"""Rebuild every artifact in dependency order.   python -m src.run_all [--skip-slow]

--skip-slow leaves TabPFN (≈2 h on CPU) and TabNet (≈1 h) as they are; their caches/artifacts are reused.
"""
from __future__ import annotations

import importlib
import sys
import time

STEPS = [
    ("dataset", "src.dataset_report"),
    ("decision_tree", "src.models.decision_tree"),
    ("catboost", "src.models.catboost_model"),
    ("umap_hdbscan", "src.models.umap_hdbscan"),
    ("fp_growth", "src.models.fpgrowth_model"),
    ("tabnet", "src.models.tabnet_model"),
    ("tabpfn", "src.models.tabpfn_model"),
    ("reference", "src.models.reference_models"),
    ("comparison", "src.comparison"),
    ("call_ranking", "src.call_ranking"),
    ("report", "src.report"),
]
SLOW = {"tabnet", "tabpfn"}


def main(argv):
    skip = "--skip-slow" in argv
    only = [a for a in argv if not a.startswith("--")]
    for name, mod in STEPS:
        if (skip and name in SLOW) or (only and name not in only):
            print(f"-- skip {name}")
            continue
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        importlib.import_module(mod).main()
        print(f"   {name} done in {time.perf_counter() - t0:.0f}s", flush=True)
    from .contract import _main
    return _main(["validate", "all"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
