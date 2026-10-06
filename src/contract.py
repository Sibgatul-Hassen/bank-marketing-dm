"""Artifact writers and schema validators (§6). The UI renders artifacts generically, so a schema
deviation breaks pages you did not write. Run before every PR:

    python -m src.contract validate <model_id>      # or: validate all
    python -m src.contract manifest                 # rebuild results/manifest.json
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import config as C

ASSUMPTION_STATUS = {"ok", "warn", "fail", "required", "unknown"}
FINDING_TYPES = {"result", "limitation", "failure"}
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "brier",
               "confusion_matrix", "train_seconds", "predict_seconds"]
EXPLAIN_KINDS = {"shap", "cluster_profile", "rule_contribution"}


# ---------------------------------------------------------------- writers

def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _clean_floats(o, ndigits=6):
    if isinstance(o, float):
        return None if not math.isfinite(o) else round(o, ndigits)
    if isinstance(o, dict):
        return {k: _clean_floats(v, ndigits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_floats(v, ndigits) for v in o]
    return o


def write_json(path: Path, obj, ndigits: int = 6) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = json.loads(json.dumps(obj, default=_default))
    path.write_text(json.dumps(_clean_floats(obj, ndigits), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def model_dir(model_id: str) -> Path:
    d = C.RESULTS_MODELS / model_id
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def write_figure(model_id_or_dir, name: str, fig) -> str:
    """Write a plotly figure as JSON. Charts are themed by the frontend, so keep them plain."""
    d = model_dir(model_id_or_dir) if isinstance(model_id_or_dir, str) else Path(model_id_or_dir)
    fig.update_layout(template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    (d / "figures").mkdir(parents=True, exist_ok=True)
    spec = _decode_typed_arrays(json.loads(fig.to_json()))
    (d / "figures" / name).write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    return name


def _decode_typed_arrays(o):
    """Plotly ≥ 6 serialises numpy arrays as base64 {dtype, bdata}; store plain, readable lists."""
    if isinstance(o, dict):
        if set(o) >= {"dtype", "bdata"}:
            import base64
            arr = np.frombuffer(base64.b64decode(o["bdata"]), dtype=np.dtype(o["dtype"]))
            if "shape" in o:
                shape = [int(x) for x in str(o["shape"]).split(",")]
                arr = arr.reshape(shape)
            if arr.dtype.kind == "f":
                arr = np.round(arr, 6)
            return _clean_floats(arr.tolist())
        return {k: _decode_typed_arrays(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode_typed_arrays(v) for v in o]
    return o


def build_metrics(model_id: str, runs: dict, y_test, protocol_extra: dict | None = None) -> dict:
    from .evaluation import PROTOCOL, baselines
    wo, w = runs["without_duration"], runs["with_duration"]
    wo["primary"], w["primary"] = True, False
    return {
        "model_id": model_id,
        "protocol": {**PROTOCOL, **(protocol_extra or {})},
        "runs": {"without_duration": wo, "with_duration": w},
        "baselines": baselines(y_test),
        "leak_delta": {k: w[k] - wo[k] for k in ("roc_auc", "pr_auc", "f1")},
    }


# ---------------------------------------------------------------- validators

class ContractError(Exception):
    pass


def _load(path: Path, errors: list) -> dict | None:
    if not path.exists():
        errors.append(f"missing {path.relative_to(C.ROOT).as_posix()}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"invalid JSON in {path.name}: {e}")
        return None


def _validate_meta(meta: dict, d: Path, errors: list):
    for k in ["model_id", "display_name", "family", "task", "one_liner", "plain_explanation",
              "year", "reference", "workflow", "edges", "formulas", "assumptions", "figures"]:
        if k not in meta:
            errors.append(f"meta.json: missing '{k}'")
    nodes = meta.get("workflow", [])
    if not 4 <= len(nodes) <= 8:
        errors.append(f"meta.json: {len(nodes)} workflow nodes (need 4–8)")
    ids = {n.get("id") for n in nodes}
    fids = {f.get("id") for f in meta.get("formulas", [])}
    for n in nodes:
        for k in ("id", "label", "detail"):
            if not n.get(k):
                errors.append(f"meta.json: node missing '{k}': {n}")
        for ref in filter(None, str(n.get("formula_ref") or "").replace(" ", "").split(",")):
            if ref not in fids:
                errors.append(f"meta.json: node {n.get('id')} references unknown formula {ref}")
    for e in meta.get("edges", []):
        if len(e) != 2 or e[0] not in ids or e[1] not in ids:
            errors.append(f"meta.json: bad edge {e}")
    for f in meta.get("formulas", []):
        if not f.get("latex") or not f.get("caption"):
            errors.append(f"meta.json: formula {f.get('id')} needs latex and caption")
    for a in meta.get("assumptions", []):
        if a.get("status") not in ASSUMPTION_STATUS:
            errors.append(f"meta.json: assumption status {a.get('status')!r} not in {ASSUMPTION_STATUS}")
    figs = meta.get("figures", [])
    if len(figs) < 3:
        errors.append(f"meta.json: {len(figs)} figures (need ≥ 3)")
    for f in figs:
        p = d / "figures" / f.get("file", "")
        if not p.exists():
            errors.append(f"figure listed but missing: {f.get('file')}")
        elif not f.get("caption"):
            errors.append(f"figure {f.get('file')} has no caption")
        elif p.suffix == ".json" and "data" not in json.loads(p.read_text(encoding="utf-8")) \
                and f.get("type") not in ("custom", "table", "points3d", "tree", "mask3d"):
            errors.append(f"figure {f.get('file')} is not plotly JSON (no 'data')")


def _validate_metrics(m: dict, errors: list):
    runs = m.get("runs", {})
    for r in ("without_duration", "with_duration"):
        run = runs.get(r)
        if not run:
            errors.append(f"metrics.json: missing run '{r}'")
            continue
        for k in METRIC_KEYS:
            if k not in run:
                errors.append(f"metrics.json: {r} missing '{k}'")
    wo = runs.get("without_duration", {})
    if wo.get("roc_auc", 0) > C.AUC_RED_FLAG:
        errors.append(f"metrics.json: ROC-AUC {wo['roc_auc']:.3f} without duration > {C.AUC_RED_FLAG} "
                      "— something leaked. Stop and audit the feature set.")
    if "baselines" not in m or "leak_delta" not in m:
        errors.append("metrics.json: needs 'baselines' and 'leak_delta'")


def _validate_explain(e: dict, errors: list):
    if e.get("kind") not in EXPLAIN_KINDS:
        errors.append(f"explain.json: kind {e.get('kind')!r} not in {EXPLAIN_KINDS}")
    n = e.get("narrative", {})
    for k in ("headline", "body", "caveat"):
        if not n.get(k):
            errors.append(f"explain.json: narrative.{k} is empty")
    if e.get("kind") == "shap":
        g = e.get("global", {})
        if not g.get("plain_text") or not g.get("features"):
            errors.append("explain.json: global.features and global.plain_text are mandatory")
        for ex in e.get("local_examples", []):
            if not ex.get("plain_text"):
                errors.append("explain.json: every local example needs plain_text")


def _validate_findings(f: dict, errors: list):
    if not f.get("headline"):
        errors.append("findings.json: missing headline")
    types = [x.get("type") for x in f.get("findings", [])]
    if any(t not in FINDING_TYPES for t in types):
        errors.append(f"findings.json: finding types must be in {FINDING_TYPES}")
    if not ({"limitation", "failure"} & set(types)):
        errors.append("findings.json: needs at least one 'limitation' or 'failure' (§6.4)")
    for k in ("comparison_note", "viva_answer"):
        if not f.get(k):
            errors.append(f"findings.json: missing '{k}'")


def validate_model_artifacts(model_id: str, raise_on_error: bool = True) -> list[str]:
    d = C.RESULTS_MODELS / model_id
    errors: list[str] = []
    meta = _load(d / "meta.json", errors)
    if meta:
        _validate_meta(meta, d, errors)
    cfg = _load(d / "config.json", errors)
    if cfg:
        for p in cfg.get("params", []):
            if not p.get("why"):
                errors.append(f"config.json: param {p.get('name')} has no 'why'")
    supervised = (meta or {}).get("task") == "supervised_classification"
    metrics = _load(d / "metrics.json", errors)
    if metrics and supervised:
        _validate_metrics(metrics, errors)
        _load(d / "curves.json", errors)
    explain = _load(d / "explain.json", errors)
    if explain:
        _validate_explain(explain, errors)
    findings = _load(d / "findings.json", errors)
    if findings:
        _validate_findings(findings, errors)
    if errors and raise_on_error:
        raise ContractError(f"{model_id}: " + "; ".join(errors))
    return errors


def validate_dataset(raise_on_error: bool = True) -> list[str]:
    errors: list[str] = []
    s = _load(C.RESULTS_DATASET / "summary.json", errors)
    q = _load(C.RESULTS_DATASET / "quality.json", errors)
    if s and s.get("sha256") != C.EXPECTED_SHA256:
        errors.append("summary.json: hash does not match config.EXPECTED_SHA256")
    if q and len(q.get("problems", [])) < 4:
        errors.append("quality.json: all four data problems must be documented")
    if errors and raise_on_error:
        raise ContractError("; ".join(errors))
    return errors


def build_manifest() -> dict:
    models = []
    for mid in C.MODEL_IDS:
        d = C.RESULTS_MODELS / mid
        entry = {"model_id": mid, "status": "missing"}
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            errs = validate_model_artifacts(mid, raise_on_error=False)
            entry.update(display_name=meta["display_name"], family=meta["family"], task=meta["task"],
                         one_liner=meta["one_liner"], status="ok" if not errs else "invalid",
                         errors=errs)
        models.append(entry)
    comp = C.RESULTS_COMPARISON
    manifest = {
        "project": "A Leakage-Aware, Visually Explained Comparison of Modern Tabular Learning Methods",
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_sha256": C.EXPECTED_SHA256,
        "dataset_ok": not validate_dataset(raise_on_error=False),
        "models": models,
        "comparison_ok": (comp / "leaderboard.json").exists(),
    }
    write_json(C.RESULTS / "manifest.json", manifest)
    return manifest


def _main(argv):
    if len(argv) >= 2 and argv[0] == "validate":
        targets = C.MODEL_IDS if argv[1] == "all" else [argv[1]]
        bad = 0
        if argv[1] in ("all", "dataset"):
            errs = validate_dataset(raise_on_error=False)
            print(f"{'FAIL' if errs else 'ok  '} dataset", *[f"\n   - {e}" for e in errs])
            bad += bool(errs)
            if argv[1] == "dataset":
                return bad
        for mid in targets:
            errs = validate_model_artifacts(mid, raise_on_error=False)
            print(f"{'FAIL' if errs else 'ok  '} {mid}", *[f"\n   - {e}" for e in errs])
            bad += bool(errs)
        return 1 if bad else 0
    if argv and argv[0] == "manifest":
        m = build_manifest()
        print(json.dumps([(x["model_id"], x["status"]) for x in m["models"]]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
