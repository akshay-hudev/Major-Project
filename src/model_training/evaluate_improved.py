"""
evaluate_improved.py
====================
Combines the per-model probabilities produced by train_improved.py into
ensembles, tunes the decision threshold on the validation set, evaluates on the
held-out test set, and prints/writes a clean before-vs-after comparison against
the baseline metrics stored in results.json.

Ensembles built:
  - mean      : simple average of calibrated per-model probabilities
  - stacked   : logistic-regression meta-learner trained on validation
                per-model probabilities (out-of-fold style: fit on val, apply
                to test)

Usage
-----
python src/model_training/evaluate_improved.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
PROB_DIR = DATA_DIR / "improved_probs"
MODELS = ["bilstm_attention", "bigru", "transformer"]


def best_f1_threshold(y, probs):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def metrics(y, probs, thr):
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


def load_probs(split, kind):
    """Return dict{model: probs} for available models."""
    out = {}
    for m in MODELS:
        p = PROB_DIR / f"{m}_{split}_{kind}.npy"
        if p.exists():
            out[m] = np.load(p)
    return out


def main():
    y_val = np.load(DATA_DIR / "y_val.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    val_cal = load_probs("val", "cal")
    test_cal = load_probs("test", "cal")
    if not test_cal:
        print("No per-model probabilities found in "
              f"{PROB_DIR}. Run train_improved.py first.")
        return
    avail = list(test_cal.keys())
    print(f"Models available for ensembling: {avail}")

    report = {"per_model": {}, "ensembles": {}}

    # ── Per-model (calibrated, tuned threshold) ────────────────────────────────
    for m in avail:
        thr = best_f1_threshold(y_val, val_cal[m])
        report["per_model"][m] = metrics(y_test, test_cal[m], thr)

    # ── Mean ensemble ──────────────────────────────────────────────────────────
    val_mean = np.mean([val_cal[m] for m in avail], axis=0)
    test_mean = np.mean([test_cal[m] for m in avail], axis=0)
    thr_mean = best_f1_threshold(y_val, val_mean)
    report["ensembles"]["mean"] = metrics(y_test, test_mean, thr_mean)

    # ── Stacked ensemble (logistic regression meta-learner) ────────────────────
    Xv = np.column_stack([val_cal[m] for m in avail])
    Xt = np.column_stack([test_cal[m] for m in avail])
    stacker = LogisticRegression(max_iter=1000, class_weight="balanced")
    stacker.fit(Xv, y_val)
    val_stack = stacker.predict_proba(Xv)[:, 1]
    test_stack = stacker.predict_proba(Xt)[:, 1]
    thr_stack = best_f1_threshold(y_val, val_stack)
    report["ensembles"]["stacked_logreg"] = metrics(y_test, test_stack, thr_stack)
    report["ensembles"]["stacked_logreg"]["coef"] = {
        m: float(c) for m, c in zip(avail, stacker.coef_[0])
    }

    # ── Baseline (from results.json) ───────────────────────────────────────────
    baseline = {}
    res_path = ROOT / "results.json"
    if res_path.exists():
        rj = json.loads(res_path.read_text())
        for m, d in rj.get("models", {}).items():
            tm = d.get("test_metrics", {})
            if tm:
                key = "bilstm_attention" if m == "bilstm_attention" else \
                      "bigru" if m == "bigru" else \
                      "transformer" if m in ("transformer", "transformer_encoder") else m
                baseline[key] = tm

    # ── Print comparison ────────────────────────────────────────────────────────
    def row(name, d):
        return (f"{name:<26} {d['accuracy']:>7.4f} {d['precision']:>7.4f} "
                f"{d['recall']:>7.4f} {d['f1']:>7.4f} {d['auc_roc']:>7.4f} "
                f"{d.get('pr_auc', float('nan')):>7.4f}")

    print("\n" + "=" * 86)
    print("  BASELINE (README / results.json, threshold 0.5)")
    print("=" * 86)
    print(f"{'model':<26} {'acc':>7} {'prec':>7} {'recall':>7} {'f1':>7} {'auc':>7} {'pr_auc':>7}")
    print("-" * 86)
    for m in MODELS:
        if m in baseline:
            b = baseline[m]
            print(f"{m:<26} {b['accuracy']:>7.4f} {b['precision']:>7.4f} "
                  f"{b['recall']:>7.4f} {b['f1_score']:>7.4f} {b['auc_roc']:>7.4f} "
                  f"{'--':>7}")

    print("\n" + "=" * 86)
    print("  IMPROVED  (calibrated probs, F1-tuned threshold from validation)")
    print("=" * 86)
    print(f"{'model':<26} {'acc':>7} {'prec':>7} {'recall':>7} {'f1':>7} {'auc':>7} {'pr_auc':>7}")
    print("-" * 86)
    for m in avail:
        print(row(m, report["per_model"][m]))
    print("-" * 86)
    for name, d in report["ensembles"].items():
        print(row("ENSEMBLE:" + name, d))
    print("=" * 86)

    # Highlight best model on each metric.
    all_improved = {**{m: report["per_model"][m] for m in avail},
                    **{"ensemble_" + k: v for k, v in report["ensembles"].items()}}
    print("\nBest improved model per metric:")
    for metric in ["accuracy", "precision", "recall", "f1", "auc_roc", "pr_auc"]:
        best = max(all_improved.items(), key=lambda kv: kv[1][metric])
        print(f"   {metric:<10}: {best[0]:<24} {best[1][metric]:.4f}")

    out = ROOT / "results_improved.json"
    payload = json.loads(out.read_text()) if out.exists() else {}
    payload["comparison"] = report
    payload["baseline"] = baseline
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved comparison -> {out}")


if __name__ == "__main__":
    main()
