"""
train_improved.py
=================
Improved training + evaluation pipeline for the MIMIC-IV cardiac-progression
task. It reuses the *exact* baseline model architectures (BiLSTM+attention,
BiGRU, Transformer encoder) so that any metric gains come from a better
TRAINING RECIPE and better POST-PROCESSING, not from changing the models.

What is different from the baseline trainers
--------------------------------------------
1. Early stopping / checkpoint selection on **validation PR-AUC** (average
   precision) instead of validation loss. The baseline monitored a
   pos_weight-inflated BCE loss, which stopped every model at epoch 5-9 of 80
   (badly undertrained). PR-AUC is the right signal for an imbalanced task.
2. **Milder, tunable class weighting.** The baseline used
   pos_weight = neg/pos (~4.9), which pushes recall high but drops precision and
   accuracy below the majority-class baseline. Default here is
   pos_weight = sqrt(neg/pos) (~2.2). Focal loss is also available.
3. **Cosine LR schedule with warmup** + AdamW weight decay, gradient clipping.
4. **Validation-based threshold tuning** (max-F1 threshold) instead of a fixed
   0.5 cutoff.
5. **Isotonic probability calibration** fit on validation.
6. Saves per-model validation/test probabilities so the models can be
   **ensembled** (see evaluate_improved.py).

Usage
-----
python src/model_training/train_improved.py --models all --epochs 80 --patience 15
python src/model_training/train_improved.py --models transformer --loss focal
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models" / "improved"
PROB_DIR = ROOT / "data" / "preprocessed" / "improved_probs"

sys.path.insert(0, str(ROOT / "src" / "model_training"))


# ══════════════════════════════════════════════════════════════════════════════
# Model architectures — mirror the baseline exactly (copied to avoid importing
# matplotlib/seaborn from the baseline trainers).
# ══════════════════════════════════════════════════════════════════════════════

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores = self.attention(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)


class LSTMHeartDiseaseModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        self.attention = AttentionLayer(self.hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.attention(out)).squeeze(-1)


class GRUHeartDiseaseModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerHeartDiseaseModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.fc(x[:, 0, :]).squeeze(-1)


MODEL_FACTORY = {
    "bilstm_attention": lambda d: LSTMHeartDiseaseModel(d, 128, 2, 0.3, True),
    "bigru":            lambda d: GRUHeartDiseaseModel(d, 128, 2, 0.3, True),
    "transformer":      lambda d: TransformerHeartDiseaseModel(d, 128, 8, 3, 0.1),
}


# ══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Binary focal loss on logits. alpha weights the positive class."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    arrays = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        p = DATA_DIR / f"{name}.npy"
        if not p.exists():
            print(f"Missing {p}. Build the dataset first "
                  f"(python src/preprocessing/build_cardiac_progression_dataset.py).")
            sys.exit(1)
        arrays[name] = np.load(p)
    return arrays


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict_probs(model, X, device, batch_size=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ══════════════════════════════════════════════════════════════════════════════

def metrics_at_threshold(y, probs, thr):
    preds = (probs >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def best_f1_threshold(y, probs):
    """Grid-search the threshold that maximizes F1 on the given set."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_one(name, data, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]
    input_size = X_tr.shape[2]

    model = MODEL_FACTORY[name](input_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ── Loss ──────────────────────────────────────────────────────────────────
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    raw_pw = neg / max(pos, 1)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        loss_desc = f"focal(alpha={args.focal_alpha}, gamma={args.focal_gamma})"
    else:
        if args.pos_weight_mode == "raw":
            pw = raw_pw
        elif args.pos_weight_mode == "sqrt":
            pw = math.sqrt(raw_pw)
        elif args.pos_weight_mode == "one":
            pw = 1.0
        else:  # fixed float
            pw = float(args.pos_weight_mode)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
        loss_desc = f"weighted_bce(pos_weight={pw:.3f}, raw={raw_pw:.3f})"

    lr = args.lr if args.lr is not None else (3e-4 if name == "transformer" else 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Cosine schedule with linear warmup (per-epoch stepping).
    warmup = max(1, args.warmup)
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)

    print(f"\n{'='*72}\n  Training [{name}]  ({n_params:,} params)")
    print(f"  loss={loss_desc}  lr={lr}  wd={args.weight_decay}  "
          f"batch={args.batch_size}  monitor=val_pr_auc")
    print(f"{'='*72}")

    best_metric = -1.0
    best_epoch = 0
    patience_ctr = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"{name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            seen += len(yb)
        scheduler.step()

        if not math.isfinite(total_loss):
            print("  Non-finite loss — aborting this model.")
            break

        val_probs = predict_probs(model, X_va, device)
        val_ap = average_precision_score(y_va, val_probs)
        val_auc = roc_auc_score(y_va, val_probs)

        marker = ""
        if val_ap > best_metric + 1e-5:
            best_metric = val_ap
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = "  <- best"
        else:
            patience_ctr += 1

        if epoch % args.log_every == 0 or marker or epoch == 1:
            print(f"  epoch {epoch:>3}/{args.epochs}  loss={total_loss/seen:.4f}  "
                  f"val_PR_AUC={val_ap:.4f}  val_AUC={val_auc:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}")

        if patience_ctr >= args.patience:
            print(f"  early stop @ epoch {epoch} (best epoch {best_epoch}, "
                  f"val_PR_AUC={best_metric:.4f})")
            break

    # ── Load best, produce calibrated probabilities ────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    val_probs = predict_probs(model, X_va, device)
    test_probs = predict_probs(model, X_te, device)

    # Isotonic calibration fit on validation.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_probs, y_va)
    val_cal = iso.predict(val_probs)
    test_cal = iso.predict(test_probs)

    # Threshold tuned on validation (calibrated), max-F1.
    thr_tuned = best_f1_threshold(y_va, val_cal)

    # Persist probabilities for ensembling.
    PROB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROB_DIR / f"{name}_val_raw.npy", val_probs)
    np.save(PROB_DIR / f"{name}_test_raw.npy", test_probs)
    np.save(PROB_DIR / f"{name}_val_cal.npy", val_cal)
    np.save(PROB_DIR / f"{name}_test_cal.npy", test_cal)

    result = {
        "params": int(n_params),
        "best_epoch": int(best_epoch),
        "val_pr_auc_best": float(best_metric),
        "loss": loss_desc,
        "lr": float(lr),
        "tuned_threshold": float(thr_tuned),
        # Baseline-style: raw probs @ 0.5 (apples-to-apples with the README).
        "test_default_0.5": metrics_at_threshold(y_te, test_probs, 0.5),
        # Calibrated probs @ tuned threshold (the improved operating point).
        "test_calibrated_tuned": metrics_at_threshold(y_te, test_cal, thr_tuned),
    }
    print(f"\n  [{name}] test @0.5 (raw):        "
          f"acc={result['test_default_0.5']['accuracy']:.4f} "
          f"F1={result['test_default_0.5']['f1']:.4f} "
          f"AUC={result['test_default_0.5']['auc_roc']:.4f} "
          f"PR_AUC={result['test_default_0.5']['pr_auc']:.4f}")
    print(f"  [{name}] test @{thr_tuned:.2f} (calibrated): "
          f"acc={result['test_calibrated_tuned']['accuracy']:.4f} "
          f"prec={result['test_calibrated_tuned']['precision']:.4f} "
          f"rec={result['test_calibrated_tuned']['recall']:.4f} "
          f"F1={result['test_calibrated_tuned']['f1']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["bilstm_attention", "bigru", "transformer"],
                    help="'all' or any of: bilstm_attention bigru transformer")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None,
                    help="Override LR (default 1e-3 for RNNs, 3e-4 for transformer)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--pos_weight_mode", default="sqrt",
                    help="raw | sqrt | one | <float>  (weighted BCE only)")
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()

    if args.models == ["all"]:
        args.models = ["bilstm_attention", "bigru", "transformer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data = load_data()
    print(f"Train {data['X_train'].shape} pos={int(data['y_train'].sum())} | "
          f"Val {data['X_val'].shape} pos={int(data['y_val'].sum())} | "
          f"Test {data['X_test'].shape} pos={int(data['y_test'].sum())}")

    results = {}
    for name in args.models:
        results[name] = train_one(name, data, args, device)

    out = ROOT / "results_improved.json"
    payload = json.loads(out.read_text()) if out.exists() else {}
    payload.update(results)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved per-model improved results -> {out}")
    print("Next: python src/model_training/evaluate_improved.py")


if __name__ == "__main__":
    main()
