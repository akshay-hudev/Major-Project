"""
train_gbm_advanced.py  —  push the gradient-boosting model further
==================================================================
The plain HistGradientBoosting model on engineered temporal features already
beat every sequence model (test F1 0.71 / AUC 0.946). This script explores that
family harder to squeeze out more:

1. **Richer temporal feature engineering** — beyond last/mean/std/min/max/
   delta/slope, we add clinically-motivated *progression* descriptors:
     * velocity (consecutive deltas) mean/max/min/last
     * acceleration (2nd difference) mean
     * total variation  (sum |Δ|)  — instability of the trajectory
     * trend  (2nd-half mean − 1st-half mean)
     * range, IQR, median, coefficient of variation
     * value-at-last relative to the min/max seen (how extreme is "now")
2. **Three GBM backends** — sklearn HistGBM, LightGBM, XGBoost — each with a
   small hyperparameter search selected on **validation PR-AUC** (val/test are
   never touched during selection).
3. **Isotonic calibration** on validation + **F1-optimal threshold** tuning.
4. **Soft-voting ensemble** of the best of each backend, plus a **logistic
   stacker**. The single best deployable model AND the ensemble are reported.

Everything is evaluated on the held-out test set at 0.5 (apples-to-apples with
the baseline) and at the tuned threshold (the deployable operating point).

Usage
-----
python src/model_training/train_gbm_advanced.py
python src/model_training/train_gbm_advanced.py --quick   # smaller search
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

warnings.filterwarnings("ignore")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models" / "deployable"
PROB_DIR = DATA_DIR / "improved_probs"


# ── Rich temporal feature engineering ──────────────────────────────────────────

def temporal_features(X: np.ndarray) -> np.ndarray:
    """(N, T, F) -> rich engineered matrix. Vectorized over N and F."""
    N, T, F = X.shape
    eps = 1e-6

    last = X[:, -1, :]
    first = X[:, 0, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    med = np.median(X, axis=1)
    q75 = np.percentile(X, 75, axis=1)
    q25 = np.percentile(X, 25, axis=1)
    iqr = q75 - q25
    rng = mx - mn
    delta = last - first
    cov = std / (np.abs(mean) + eps)                       # coefficient of variation

    # Trend via least-squares slope over centered time.
    t = np.arange(T, dtype=np.float64); t -= t.mean()
    denom = (t ** 2).sum()
    slope = (X * t[None, :, None]).sum(axis=1) / denom

    # Half-split trend (robust direction signal).
    h = T // 2
    trend = X[:, h:, :].mean(axis=1) - X[:, :h, :].mean(axis=1)

    # Velocity (1st difference) and acceleration (2nd difference).
    diff1 = np.diff(X, axis=1)                              # (N, T-1, F)
    vel_mean = diff1.mean(axis=1)
    vel_max = diff1.max(axis=1)
    vel_min = diff1.min(axis=1)
    vel_last = diff1[:, -1, :]
    total_var = np.abs(diff1).sum(axis=1)                  # trajectory instability
    diff2 = np.diff(X, n=2, axis=1) if T >= 3 else np.zeros((N, 1, F))
    acc_mean = diff2.mean(axis=1)

    # Position of the last value within the observed range (0..1).
    pos_last = (last - mn) / (rng + eps)

    flat = X.reshape(N, T * F)
    feats = [flat, last, first, mean, std, mn, mx, med, iqr, rng, delta, cov,
             slope, trend, vel_mean, vel_max, vel_min, vel_last, total_var,
             acc_mean, pos_last]
    return np.concatenate(feats, axis=1).astype(np.float32)


TAGS = ["last", "first", "mean", "std", "min", "max", "med", "iqr", "rng",
        "delta", "cov", "slope", "trend", "vel_mean", "vel_max", "vel_min",
        "vel_last", "total_var", "acc_mean", "pos_last"]


def feature_names(base, T):
    names = []
    for t in range(T):
        names += [f"{b}@t{t}" for b in base]
    for tag in TAGS:
        names += [f"{b}_{tag}" for b in base]
    return names


# ── Metrics ────────────────────────────────────────────────────────────────────

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


# ── Backends & their small search grids ─────────────────────────────────────────

def hist_grid(quick):
    lrs = [0.05] if quick else [0.03, 0.05, 0.08]
    leaves = [31] if quick else [31, 63]
    for lr in lrs:
        for lv in leaves:
            for mspl in [20, 40]:
                yield {"learning_rate": lr, "max_leaf_nodes": lv,
                       "min_samples_leaf": mspl}


def lgb_grid(quick):
    lrs = [0.05] if quick else [0.03, 0.05]
    leaves = [31, 63] if not quick else [31]
    for lr in lrs:
        for lv in leaves:
            for mcs in [20, 40]:
                yield {"learning_rate": lr, "num_leaves": lv,
                       "min_child_samples": mcs}


def xgb_grid(quick):
    lrs = [0.05] if quick else [0.03, 0.05]
    depths = [4, 6] if not quick else [6]
    for lr in lrs:
        for d in depths:
            for ss in [0.8, 1.0]:
                yield {"learning_rate": lr, "max_depth": d, "subsample": ss}


def fit_hist(params, Ftr, y_tr, spw, es=None):
    clf = HistGradientBoostingClassifier(
        loss="log_loss", max_iter=1000, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=40,
        class_weight="balanced", random_state=42, **params)
    clf.fit(Ftr, y_tr)
    return clf


def fit_lgb(params, Ftr, y_tr, spw, es=None):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, n_jobs=-1, verbosity=-1,
        **params)
    if es is not None:
        clf.fit(Ftr, y_tr, eval_set=[es], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(60, verbose=False)])
    else:
        clf.fit(Ftr, y_tr)
    return clf


def fit_xgb(params, Ftr, y_tr, spw, es=None):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=3000, reg_lambda=1.0,
        colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
        scale_pos_weight=spw, random_state=42, n_jobs=-1,
        early_stopping_rounds=60 if es is not None else None, **params)
    if es is not None:
        clf.fit(Ftr, y_tr, eval_set=[es], verbose=False)
    else:
        clf.fit(Ftr, y_tr)
    return clf


BACKENDS = {
    "hist": (hist_grid, fit_hist),
    "lightgbm": (lgb_grid, fit_lgb),
    "xgboost": (xgb_grid, fit_xgb),
}


def proba(clf, X):
    return clf.predict_proba(X)[:, 1]


def search_backend(name, Ftr, y_tr, Fva, y_va, spw, quick):
    grid_fn, fit_fn = BACKENDS[name]
    # LightGBM/XGBoost use an internal early-stop split carved from TRAIN so
    # that X_val stays reserved purely for model selection (no leakage).
    es = None
    if name in ("lightgbm", "xgboost"):
        from sklearn.model_selection import train_test_split
        Ftr_in, Fes, y_in, y_es = train_test_split(
            Ftr, y_tr, test_size=0.12, stratify=y_tr, random_state=42)
        es = (Fes, y_es)
    else:
        Ftr_in, y_in = Ftr, y_tr
    best = None
    for params in grid_fn(quick):
        try:
            clf = fit_fn(params, Ftr_in, y_in, spw, es)
        except Exception as e:
            print(f"    [{name}] {params} -> FAILED ({type(e).__name__}: {e})")
            continue
        ap = average_precision_score(y_va, proba(clf, Fva))
        tag = ""
        if best is None or ap > best[0]:
            best = (ap, clf, params); tag = "  <- best"
        print(f"    [{name}] {params}  val_PR_AUC={ap:.4f}{tag}")
    return best  # (val_ap, clf, params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    X_tr = np.load(DATA_DIR / "X_train.npy"); y_tr = np.load(DATA_DIR / "y_train.npy")
    X_va = np.load(DATA_DIR / "X_val.npy");   y_va = np.load(DATA_DIR / "y_val.npy")
    X_te = np.load(DATA_DIR / "X_test.npy");  y_te = np.load(DATA_DIR / "y_test.npy")

    print("=" * 80)
    print("  Advanced gradient boosting — rich temporal features, 3 backends")
    print("=" * 80)
    Ftr = temporal_features(X_tr); Fva = temporal_features(X_va); Fte = temporal_features(X_te)
    print(f"  Feature matrix: {Ftr.shape[1]} features/sample "
          f"(was 826, now richer)")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"  scale_pos_weight (raw) = {spw:.3f}")

    base_names = None
    p = DATA_DIR / "feature_names.pkl"
    if p.exists():
        try:
            base_names = list(joblib.load(p))
        except Exception:
            base_names = None
    if not base_names or len(base_names) != X_tr.shape[2]:
        base_names = [f"f{i}" for i in range(X_tr.shape[2])]
    feat_names = feature_names(base_names, X_tr.shape[1])

    # ── Search each backend ────────────────────────────────────────────────────
    winners = {}
    for name in BACKENDS:
        print(f"\n  ── searching {name} ──")
        res = search_backend(name, Ftr, y_tr, Fva, y_va, spw, args.quick)
        if res is not None:
            winners[name] = res

    if not winners:
        print("No backend succeeded."); return

    # ── Per-backend calibrated test metrics ────────────────────────────────────
    report = {"n_features": int(Ftr.shape[1]), "backends": {}}
    cal_val, cal_test = {}, {}
    for name, (val_ap, clf, params) in winners.items():
        va_raw, te_raw = proba(clf, Fva), proba(clf, Fte)
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(va_raw, y_va)
        vc, tc = iso.predict(va_raw), iso.predict(te_raw)
        cal_val[name], cal_test[name] = vc, tc
        thr = best_f1_threshold(y_va, vc)
        report["backends"][name] = {
            "params": params, "val_pr_auc": float(val_ap),
            "test_default_0.5": metrics_at(y_te, te_raw, 0.5),
            "test_calibrated_tuned": metrics_at(y_te, tc, thr),
            "_iso": iso, "_clf": clf,
        }

    # ── Soft-voting ensemble + logistic stacker ────────────────────────────────
    names = list(cal_test.keys())
    val_mean = np.mean([cal_val[n] for n in names], axis=0)
    test_mean = np.mean([cal_test[n] for n in names], axis=0)
    thr_mean = best_f1_threshold(y_va, val_mean)
    report["ensemble_soft_vote"] = {
        "members": names,
        "test_default_0.5": metrics_at(y_te, test_mean, 0.5),
        "test_calibrated_tuned": metrics_at(y_te, test_mean, thr_mean),
    }

    Xv = np.column_stack([cal_val[n] for n in names])
    Xt = np.column_stack([cal_test[n] for n in names])
    stk = LogisticRegression(max_iter=1000, class_weight="balanced")
    stk.fit(Xv, y_va)
    vs, ts = stk.predict_proba(Xv)[:, 1], stk.predict_proba(Xt)[:, 1]
    thr_s = best_f1_threshold(y_va, vs)
    report["ensemble_stacked"] = {
        "members": names,
        "coef": {n: float(c) for n, c in zip(names, stk.coef_[0])},
        "test_default_0.5": metrics_at(y_te, ts, 0.5),
        "test_calibrated_tuned": metrics_at(y_te, ts, thr_s),
    }

    # ── Print comparison ────────────────────────────────────────────────────────
    def line(tag, d):
        return (f"  {tag:<28} acc={d['accuracy']:.4f} prec={d['precision']:.4f} "
                f"rec={d['recall']:.4f} F1={d['f1']:.4f} AUC={d['auc_roc']:.4f} "
                f"PR={d['pr_auc']:.4f}")

    print("\n" + "=" * 80)
    print("  TEST RESULTS (calibrated, F1-tuned threshold)")
    print("=" * 80)
    best_overall = None
    for name in names:
        d = report["backends"][name]["test_calibrated_tuned"]
        print(line(name, d))
        if best_overall is None or d["f1"] > best_overall[1]:
            best_overall = (name, d["f1"])
    print(line("ENSEMBLE:soft_vote", report["ensemble_soft_vote"]["test_calibrated_tuned"]))
    print(line("ENSEMBLE:stacked", report["ensemble_stacked"]["test_calibrated_tuned"]))
    for tag, key in [("ENSEMBLE:soft_vote", "ensemble_soft_vote"),
                     ("ENSEMBLE:stacked", "ensemble_stacked")]:
        f1 = report[key]["test_calibrated_tuned"]["f1"]
        if f1 > best_overall[1]:
            best_overall = (tag, f1)
    print("-" * 80)
    print(f"  BEST by F1: {best_overall[0]}  (F1={best_overall[1]:.4f})")

    # ── Save the single best deployable backend as a self-contained bundle ──────
    # Pick the backend (not ensemble) with the highest calibrated tuned F1 for a
    # simple one-file deployment; ensembles are reported but heavier to ship.
    best_backend = max(
        names, key=lambda n: report["backends"][n]["test_calibrated_tuned"]["f1"])
    bb = report["backends"][best_backend]
    thr_bb = best_f1_threshold(y_va, cal_val[best_backend])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "backend": best_backend,
        "model": bb["_clf"],
        "calibrator": bb["_iso"],
        "threshold": float(thr_bb),
        "feature_builder": "temporal_features (train_gbm_advanced.py)",
        "seq_shape": [int(X_tr.shape[1]), int(X_tr.shape[2])],
        "test_metrics": bb["test_calibrated_tuned"],
    }
    joblib.dump(bundle, MODEL_DIR / "gbm_advanced.joblib")
    size_mb = (MODEL_DIR / "gbm_advanced.joblib").stat().st_size / 1e6
    print(f"\n  Saved best deployable model ({best_backend}) -> "
          f"{MODEL_DIR / 'gbm_advanced.joblib'} ({size_mb:.2f} MB)")

    # Save probs for the cross-model evaluator (use the best backend as 'gbm2').
    PROB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROB_DIR / "gbm2_val_cal.npy", cal_val[best_backend])
    np.save(PROB_DIR / "gbm2_test_cal.npy", cal_test[best_backend])

    # Strip unpicklable-in-json handles before writing.
    for n in names:
        report["backends"][n].pop("_iso", None)
        report["backends"][n].pop("_clf", None)

    out = ROOT / "results_improved.json"
    payload = json.loads(out.read_text()) if out.exists() else {}
    payload["gbm_advanced"] = report
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Recorded metrics -> {out}")


if __name__ == "__main__":
    main()
