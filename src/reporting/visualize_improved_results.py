"""
visualize_improved_results.py  —  result visualizations for the Aug 2026 run
============================================================================
Renders the "data visualizations of results" that accompany the improved
models (gradient-boosted trees on engineered temporal features vs the sequence
baselines). All figures are written to ``docs/assets/results/`` as PNGs so they
render inline in the README on GitHub.

It reads only *already-computed* artifacts:
  * ``data/preprocessed/y_test.npy`` / ``y_val.npy``            (labels)
  * ``data/preprocessed/improved_probs/*_test_cal.npy``          (calibrated probs)
  * ``results_baseline_reproduced.json`` / ``results_improved.json`` (metrics)

Nothing here touches raw MIMIC-IV records; it only visualizes model outputs.

Figures produced
----------------
  improved_model_comparison.png   best baseline vs best GBM across 5 metrics
  improved_roc_curves.png         ROC for every model with calibrated probs
  improved_pr_curves.png          precision-recall curves + positive base rate
  improved_confusion_matrix.png   confusion matrix of the best GBM ensemble
  improved_feature_importance.png top engineered temporal features (LightGBM gain)

Usage
-----
python src/reporting/visualize_improved_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 and crash on the box-drawing / unit glyphs
# used below; force UTF-8 before the first print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
PROB_DIR = DATA_DIR / "improved_probs"
OUT_DIR = ROOT / "docs" / "assets" / "results"

# Calibrated test-set probabilities we know how to plot, in display order.
# label -> (prob filename, colour, is_gbm)
CURVE_MODELS = [
    ("BiLSTM-Attention", "bilstm_attention_test_cal.npy", "#7f8c8d", False),
    ("BiGRU",            "bigru_test_cal.npy",            "#95a5a6", False),
    ("Transformer",      "transformer_test_cal.npy",      "#2980b9", False),
    ("GBM (simple, 826 feats)",   "gbm_test_cal.npy",     "#27ae60", True),
    ("GBM (advanced, HistGB)",    "gbm2_test_cal.npy",    "#c0392b", True),
]


def _load_probs(y):
    """Return [(label, probs, colour, is_gbm)] for every prob file present."""
    out = []
    for label, fname, colour, is_gbm in CURVE_MODELS:
        p = PROB_DIR / fname
        if not p.exists():
            print(f"  (skip {label}: {fname} not found)")
            continue
        probs = np.load(p)
        if probs.shape[0] != y.shape[0]:
            print(f"  (skip {label}: {probs.shape[0]} probs != {y.shape[0]} labels)")
            continue
        out.append((label, probs, colour, is_gbm))
    return out


def fig_comparison_bars():
    """Grouped bars: best reproduced baseline vs best improved GBM ensemble."""
    base = json.loads((ROOT / "results_baseline_reproduced.json").read_text())["transformer"]
    gbm = json.loads((ROOT / "results_improved.json").read_text())
    ens = gbm["gbm_advanced"]["ensemble_soft_vote"]["test_calibrated_tuned"]

    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    base_vals = [base[m] for m in metrics]
    gbm_vals = [ens[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x - w / 2, base_vals, w, label="Best baseline (Transformer)", color="#2980b9")
    b2 = ax.bar(x + w / 2, gbm_vals, w, label="Best improved (GBM soft-vote ensemble)", color="#c0392b")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.3f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Score")
    ax.set_title("MIMIC-IV cardiac progression — baseline vs improved (held-out test set)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "improved_model_comparison.png")


def fig_roc(models, y):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for label, probs, colour, is_gbm in models:
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        ax.plot(fpr, tpr, color=colour, lw=2.4 if is_gbm else 1.6,
                label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — improved GBM vs sequence baselines")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "improved_roc_curves.png")


def fig_pr(models, y):
    base_rate = float(y.mean())
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for label, probs, colour, is_gbm in models:
        prec, rec, _ = precision_recall_curve(y, probs)
        ap = average_precision_score(y, probs)
        ax.plot(rec, prec, color=colour, lw=2.4 if is_gbm else 1.6,
                label=f"{label} (AP={ap:.3f})")
    ax.axhline(base_rate, color="k", ls="--", lw=1, alpha=0.5,
               label=f"positive base rate = {base_rate:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves — improved GBM vs sequence baselines")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, "improved_pr_curves.png")


def fig_confusion():
    d = json.loads((ROOT / "results_improved.json").read_text())
    ct = d["gbm_advanced"]["ensemble_soft_vote"]["test_calibrated_tuned"]
    cm = np.array(ct["confusion_matrix"])
    thr = ct["threshold"]
    fig, ax = plt.subplots(figsize=(5.4, 5))
    im = ax.imshow(cm, cmap="Reds")
    classes = ["Stable/Improving", "Worsening"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / total:.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_title(f"GBM soft-vote ensemble — confusion matrix\n(test set, tuned threshold = {thr})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "improved_confusion_matrix.png")


def fig_feature_importance():
    """Top engineered temporal features by LightGBM gain (fast, native importances)."""
    try:
        import lightgbm as lgb
        import joblib
        sys.path.insert(0, str(ROOT / "src" / "model_training"))
        from train_gbm_advanced import temporal_features, feature_names
    except Exception as e:
        print(f"  (skip feature importance: {type(e).__name__}: {e})")
        return

    X_tr = np.load(DATA_DIR / "X_train.npy"); y_tr = np.load(DATA_DIR / "y_train.npy")
    Ftr = temporal_features(X_tr)
    p = DATA_DIR / "feature_names.pkl"
    base = list(joblib.load(p)) if p.exists() else [f"f{i}" for i in range(X_tr.shape[2])]
    if len(base) != X_tr.shape[2]:
        base = [f"f{i}" for i in range(X_tr.shape[2])]
    names = feature_names(base, X_tr.shape[1])
    if len(names) != Ftr.shape[1]:
        names = [f"feat_{i}" for i in range(Ftr.shape[1])]

    from sklearn.model_selection import train_test_split
    Xin, Xes, yin, yes = train_test_split(Ftr, y_tr, test_size=0.12, stratify=y_tr, random_state=42)
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = lgb.LGBMClassifier(objective="binary", n_estimators=2000, learning_rate=0.05,
                             num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                             random_state=42, n_jobs=-1, verbosity=-1)
    clf.fit(Xin, yin, eval_set=[(Xes, yes)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(60, verbose=False)])

    imp = clf.feature_importances_.astype(float)
    order = np.argsort(imp)[::-1][:20][::-1]  # top 20, ascending for barh
    top_names = [names[i] for i in order]
    top_vals = imp[order]

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.barh(range(len(order)), top_vals, color="#c0392b")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("LightGBM gain importance")
    ax.set_title("Top 20 engineered temporal features (drivers of the GBM)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "improved_feature_importance.png")


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


def main():
    print("=" * 74)
    print("  Rendering improved-result visualizations -> docs/assets/results/")
    print("=" * 74)
    y = np.load(DATA_DIR / "y_test.npy")
    models = _load_probs(y)

    for fn, args in [(fig_comparison_bars, ()), (fig_roc, (models, y)),
                     (fig_pr, (models, y)), (fig_confusion, ()),
                     (fig_feature_importance, ())]:
        try:
            fn(*args)
        except Exception as e:  # one bad figure shouldn't sink the rest
            print(f"  [WARN] {fn.__name__} failed: {type(e).__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
