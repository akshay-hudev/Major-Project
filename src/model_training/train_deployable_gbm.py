"""
train_deployable_gbm.py
=======================
A production-oriented alternative to the sequence models. Instead of a larger
neural network, this trains a **gradient-boosted decision tree** ensemble
(sklearn's HistGradientBoostingClassifier — a LightGBM-style histogram GBM) on
*engineered temporal features* derived from the (6, 59) input sequences.

Why this model for this project
-------------------------------
* Gradient-boosted trees are state-of-the-art on tabular / temporal-tabular
  clinical data and are naturally robust to class imbalance.
* Trains in ~1-2 minutes on CPU (no GPU), versus ~10 min/model for the RNNs.
* Deployable: a single small artifact (joblib), millisecond inference, no
  PyTorch runtime, native feature importances for clinical interpretability.

Pipeline
--------
1. Temporal feature engineering: for each of the 59 raw features we compute
   last, first, mean, std, min, max, delta(last-first) and slope over the 6
   ICU timesteps, and also keep the flattened raw sequence. (progression =
   delta / slope; state = last; volatility = std.)
2. HistGradientBoostingClassifier with class_weight="balanced" and internal
   early stopping (validation carved from TRAIN only — our val/test stay clean).
3. Isotonic calibration fit on the validation set.
4. Decision threshold tuned for max-F1 on validation.
5. Evaluate on the held-out test set at 0.5 (apples-to-apples with the baseline)
   and at the tuned threshold (the deployable operating point).

Usage
-----
python src/model_training/train_deployable_gbm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

# Windows console defaults to cp1252 and crashes on the box-drawing/emoji prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models" / "deployable"


# ── Feature engineering ────────────────────────────────────────────────────────

def temporal_features(X: np.ndarray) -> np.ndarray:
    """(N, T, F) sequences -> (N, F*(T+7)) engineered feature matrix."""
    N, T, F = X.shape
    last = X[:, -1, :]
    first = X[:, 0, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    delta = last - first                       # net change across the stay(s)
    t = np.arange(T, dtype=np.float64)
    t -= t.mean()
    denom = (t ** 2).sum()
    slope = (X * t[None, :, None]).sum(axis=1) / denom   # linear trend per feature
    flat = X.reshape(N, T * F)                 # every timestep x feature
    return np.concatenate(
        [flat, last, first, mean, std, mn, mx, delta, slope], axis=1
    ).astype(np.float32)


def feature_names(base_names, T):
    names = []
    for t in range(T):
        names += [f"{b}@t{t}" for b in base_names]
    for tag in ["last", "first", "mean", "std", "min", "max", "delta", "slope"]:
        names += [f"{b}_{tag}" for b in base_names]
    return names


# ── Metrics ──────────────────────────────────────────────────────────────────

def metrics_at(y, probs, thr):
    preds = (probs >= thr).astype(int)
    return {
        "threshold": round(float(thr), 4),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def best_f1_threshold(y, probs):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def main():
    X_tr = np.load(DATA_DIR / "X_train.npy"); y_tr = np.load(DATA_DIR / "y_train.npy")
    X_va = np.load(DATA_DIR / "X_val.npy");   y_va = np.load(DATA_DIR / "y_val.npy")
    X_te = np.load(DATA_DIR / "X_test.npy");  y_te = np.load(DATA_DIR / "y_test.npy")

    print("=" * 78)
    print("  Deployable model: HistGradientBoosting on engineered temporal features")
    print("=" * 78)
    print(f"  Train {X_tr.shape} pos={int(y_tr.sum())} | "
          f"Val {X_va.shape} pos={int(y_va.sum())} | "
          f"Test {X_te.shape} pos={int(y_te.sum())}")

    Ftr = temporal_features(X_tr)
    Fva = temporal_features(X_va)
    Fte = temporal_features(X_te)
    print(f"  Engineered feature matrix: {Ftr.shape[1]} features/sample")

    base_names = None
    fn_path = DATA_DIR / "feature_names.pkl"
    if fn_path.exists():
        try:
            base_names = list(joblib.load(fn_path))
        except Exception:
            base_names = None
    if not base_names or len(base_names) != X_tr.shape[2]:
        base_names = [f"f{i}" for i in range(X_tr.shape[2])]
    feat_names = feature_names(base_names, X_tr.shape[1])

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=800,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,          # carves its own val slice from TRAIN
        validation_fraction=0.12,
        n_iter_no_change=30,
        class_weight="balanced",
        random_state=42,
    )
    print("\n  Training gradient-boosted trees (early stopping on internal val)...")
    clf.fit(Ftr, y_tr)
    print(f"  Trees built: {clf.n_iter_}  (max 800)")

    va_raw = clf.predict_proba(Fva)[:, 1]
    te_raw = clf.predict_proba(Fte)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(va_raw, y_va)
    va_cal = iso.predict(va_raw)
    te_cal = iso.predict(te_raw)

    thr = best_f1_threshold(y_va, va_cal)

    res = {
        "params_trees": int(clf.n_iter_),
        "n_features": int(Ftr.shape[1]),
        "tuned_threshold": float(thr),
        "test_default_0.5": metrics_at(y_te, te_raw, 0.5),
        "test_calibrated_tuned": metrics_at(y_te, te_cal, thr),
    }

    d05 = res["test_default_0.5"]
    dtu = res["test_calibrated_tuned"]
    print("\n  TEST @0.5 (raw):        "
          f"acc={d05['accuracy']:.4f}  prec={d05['precision']:.4f}  "
          f"rec={d05['recall']:.4f}  F1={d05['f1']:.4f}  "
          f"AUC={d05['auc_roc']:.4f}  PR_AUC={d05['pr_auc']:.4f}")
    print(f"  TEST @{thr:.2f} (calibrated): "
          f"acc={dtu['accuracy']:.4f}  prec={dtu['precision']:.4f}  "
          f"rec={dtu['recall']:.4f}  F1={dtu['f1']:.4f}  "
          f"AUC={dtu['auc_roc']:.4f}  PR_AUC={dtu['pr_auc']:.4f}")

    # Top features by permutation importance (small sample for speed).
    try:
        from sklearn.inspection import permutation_importance
        idx = np.random.RandomState(42).choice(len(Fva), size=min(1500, len(Fva)),
                                                replace=False)
        pi = permutation_importance(clf, Fva[idx], y_va[idx], n_repeats=5,
                                    random_state=42, scoring="average_precision")
        order = np.argsort(pi.importances_mean)[::-1][:15]
        res["top_features"] = [
            {"feature": feat_names[i], "importance": float(pi.importances_mean[i])}
            for i in order
        ]
        print("\n  Top 15 features (permutation importance, PR-AUC drop):")
        for i in order:
            print(f"    {feat_names[i]:<28} {pi.importances_mean[i]:.4f}")
    except Exception as e:
        print(f"  (permutation importance skipped: {e})")

    # Persist a self-contained deployable bundle.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": clf,
        "calibrator": iso,
        "threshold": float(thr),
        "feature_builder": "temporal_features (see train_deployable_gbm.py)",
        "seq_shape": [int(X_tr.shape[1]), int(X_tr.shape[2])],
        "metrics_test": res,
    }
    joblib.dump(bundle, MODEL_DIR / "gbm_temporal.joblib")
    size_mb = (MODEL_DIR / "gbm_temporal.joblib").stat().st_size / 1e6
    print(f"\n  Saved deployable bundle -> {MODEL_DIR / 'gbm_temporal.joblib'} "
          f"({size_mb:.2f} MB)")

    # Save probabilities so this model can join the deep-model ensemble.
    prob_dir = DATA_DIR / "improved_probs"
    prob_dir.mkdir(parents=True, exist_ok=True)
    np.save(prob_dir / "gbm_val_cal.npy", va_cal)
    np.save(prob_dir / "gbm_test_cal.npy", te_cal)

    out = ROOT / "results_improved.json"
    payload = json.loads(out.read_text()) if out.exists() else {}
    payload["gbm_temporal"] = res
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Recorded metrics -> {out}")


if __name__ == "__main__":
    main()
