"""
train_gtan.py  —  Gated Temporal Attention Network (production-grade)
====================================================================
A Temporal-Fusion-Transformer-inspired architecture purpose-built for the
MIMIC-IV cardiac-progression task (short multivariate clinical sequences,
shape (6, 59), ~16.9% positive). It is designed to beat the baseline
BiLSTM/BiGRU/Transformer AND be suitable for real-world deployment
(interpretable feature selection, calibrated probabilities, small model).

Architecture (GatedTemporalAttentionNet)
-----------------------------------------
1. Per-timestep **Variable Selection Network (VSN)**: a Gated Residual Network
   (GRN) produces a softmax weight over the 59 input variables at each timestep,
   so the model learns *which* vitals/labs matter and when. The averaged
   selection weights are a clinically-meaningful explanation of the model.
2. Feature transform: each variable is passed through its own GRN before being
   combined by the selection weights -> a d_model context vector per timestep.
3. **BiLSTM local encoder** captures short-range temporal dynamics, wrapped in a
   GLU gate + add & norm (TFT-style locality enhancement).
4. **Multi-head self-attention** over the encoded sequence for long-range
   interactions, again gated + residual + norm.
5. **Attention pooling** with a learned query collapses the sequence to one
   vector; a final GRN + linear head produces the logit.

All sub-blocks use Gated Residual Networks (GLU gating + residual + LayerNorm),
which is the regularised nonlinearity that makes TFT robust on small/medium
tabular-temporal data.

Production training recipe
--------------------------
* Focal loss (handles imbalance without the recall/precision blowout of a large
  pos_weight) — falls back to weighted BCE via --loss bce.
* AdamW + linear-warmup cosine schedule, gradient clipping.
* Early stopping / checkpoint on **validation PR-AUC** (average precision).
* **Stochastic Weight Averaging (SWA)** over the tail of training for a flatter,
  better-generalising minimum.
* **Deep ensemble**: train N seeds and average calibrated probabilities.
* Isotonic calibration on validation + F1-optimal threshold tuning.

Device-agnostic: uses CUDA automatically if available (e.g. on Colab), else CPU.
On this data (6x59, ~30k rows) CPU is only minutes per seed.

Usage
-----
python src/model_training/train_gtan.py --seeds 5 --epochs 120 --loss focal
python src/model_training/train_gtan.py --seeds 1 --epochs 40    # quick probe
"""

from __future__ import annotations

import argparse
import copy
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

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "preprocessed"
MODEL_DIR = ROOT / "models" / "gtan"
PROB_DIR = DATA_DIR / "improved_probs"


# ══════════════════════════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════════════════════════

class GLU(nn.Module):
    """Gated Linear Unit: a * sigmoid(b)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out * 2)
        self.d_out = d_out

    def forward(self, x):
        x = self.fc(x)
        a, b = x[..., :self.d_out], x[..., self.d_out:]
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    """Gated Residual Network (Temporal Fusion Transformer)."""
    def __init__(self, d_in, d_hidden, d_out=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.fc2(h)
        h = self.drop(h)
        return self.norm(self.skip(x) + self.glu(h))


class VariableSelection(nn.Module):
    """Per-timestep softmax selection over input variables.

    Each variable gets its OWN linear embedding (1 -> d_model), giving it a
    distinct identity, then a learned softmax weight (from a GRN over all
    variables) selects/combines them into a per-timestep context which a shared
    GRN refines. Fully vectorized over the variable dimension — no Python loop —
    so it runs ~50x faster on CPU than a per-variable nn.Module list while
    preserving the interpretable selection weights.
    """
    def __init__(self, n_vars, d_model, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        # Selection weights: GRN over all variables -> softmax over variables.
        self.weight_grn = GRN(n_vars, d_model, n_vars, dropout)
        # Per-variable linear embedding (variable v: value * W[v] + b[v]).
        self.emb_w = nn.Parameter(torch.empty(n_vars, d_model))
        self.emb_b = nn.Parameter(torch.zeros(n_vars, d_model))
        nn.init.xavier_uniform_(self.emb_w)
        # Shared GRN refines the selected context.
        self.ctx_grn = GRN(d_model, d_model, d_model, dropout)

    def forward(self, x):                       # x: (B, T, n_vars)
        w = torch.softmax(self.weight_grn(x), dim=-1)          # (B, T, n_vars)
        # Per-variable embedding via broadcast: (B,T,V,1)*(V,d) -> (B,T,V,d).
        emb = x.unsqueeze(-1) * self.emb_w + self.emb_b        # (B, T, V, d)
        ctx = (w.unsqueeze(-1) * emb).sum(dim=-2)              # (B, T, d)
        return self.ctx_grn(ctx), w


class GatedTemporalAttentionNet(nn.Module):
    def __init__(self, n_vars, d_model=64, lstm_layers=1, nhead=4, dropout=0.1):
        super().__init__()
        self.vsn = VariableSelection(n_vars, d_model, dropout)
        self.lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_glu = GLU(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.enrich = GRN(d_model, d_model, d_model, dropout)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.attn_glu = GLU(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Attention pooling with a learned query.
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                               batch_first=True)
        self.head_grn = GRN(d_model, d_model, d_model, dropout)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x, return_weights=False):
        ctx, vsel = self.vsn(x)                               # (B, T, d)
        lstm_out, _ = self.lstm(ctx)                          # (B, T, 2d)
        local = self.norm1(ctx + self.lstm_glu(lstm_out))     # gated residual
        enriched = self.enrich(local)
        att, _ = self.attn(enriched, enriched, enriched)
        seq = self.norm2(enriched + self.attn_glu(att))       # (B, T, d)
        q = self.query.expand(seq.size(0), -1, -1)
        pooled, pool_w = self.pool_attn(q, seq, seq)          # (B, 1, d)
        pooled = pooled.squeeze(1)
        logit = self.out(self.head_grn(pooled)).squeeze(-1)
        if return_weights:
            return logit, vsel, pool_w
        return logit


# ══════════════════════════════════════════════════════════════════════════════
# Loss / metrics / data
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()


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


def load_arrays():
    a = {}
    for n in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        a[n] = np.load(DATA_DIR / f"{n}.npy")
    return a


@torch.no_grad()
def predict_probs(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════════════
# Train one seed
# ══════════════════════════════════════════════════════════════════════════════

def train_seed(seed, data, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]
    n_vars = X_tr.shape[2]

    model = GatedTemporalAttentionNet(
        n_vars, d_model=args.d_model, lstm_layers=args.lstm_layers,
        nhead=args.nhead, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.loss == "focal":
        criterion = FocalLoss(args.focal_alpha, args.focal_gamma)
        loss_desc = f"focal(a={args.focal_alpha},g={args.focal_gamma})"
    else:
        neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
        pw = math.sqrt(neg / max(pos, 1))
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
        loss_desc = f"weighted_bce(pw={pw:.2f})"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    warmup = max(1, args.warmup)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True, num_workers=0)

    print(f"\n  [seed {seed}] {loss_desc}  params={n_params:,}  "
          f"d_model={args.d_model}  device={device}")

    best_ap, best_epoch, patience_ctr = -1.0, 0, 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_DIR / f"gtan_seed{seed}_best.pt"

    swa_state, swa_n = None, 0
    swa_start = int(args.epochs * args.swa_start_frac)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(yb)
        sched.step()

        # SWA accumulation over the training tail.
        if epoch >= swa_start:
            sd = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.clone().float() for k, v in sd.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k in swa_state:
                    swa_state[k] += (sd[k].float() - swa_state[k]) / swa_n

        val_ap = average_precision_score(y_va, predict_probs(model, X_va, device))
        marker = ""
        if val_ap > best_ap + 1e-5:
            best_ap, best_epoch, patience_ctr = val_ap, epoch, 0
            torch.save(model.state_dict(), ckpt)
            marker = " <- best"
        else:
            patience_ctr += 1

        if epoch % args.log_every == 0 or marker or epoch == 1:
            print(f"    ep {epoch:>3}/{args.epochs}  loss={run/len(X_tr):.4f}  "
                  f"val_PR_AUC={val_ap:.4f}  lr={sched.get_last_lr()[0]:.2e}{marker}")

        if patience_ctr >= args.patience:
            print(f"    early stop @ {epoch} (best {best_epoch}, PR_AUC={best_ap:.4f})")
            break

    # Evaluate SWA weights; keep whichever (best-checkpoint vs SWA) wins on val PR-AUC.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    best_val_probs = predict_probs(model, X_va, device)
    best_ap_final = average_precision_score(y_va, best_val_probs)

    if swa_state is not None:
        swa_model = copy.deepcopy(model)
        swa_model.load_state_dict(swa_state)
        swa_val_probs = predict_probs(swa_model, X_va, device)
        swa_ap = average_precision_score(y_va, swa_val_probs)
        print(f"    seed {seed}: best-ckpt val PR_AUC={best_ap_final:.4f} | "
              f"SWA val PR_AUC={swa_ap:.4f}")
        if swa_ap > best_ap_final:
            model = swa_model
            torch.save(swa_state, MODEL_DIR / f"gtan_seed{seed}_swa.pt")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Main: deep ensemble
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--loss", choices=["focal", "bce"], default="focal")
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--swa_start_frac", type=float, default=0.75)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_arrays()

    print("=" * 78)
    print("  Gated Temporal Attention Network (TFT-style) — deep ensemble")
    print("=" * 78)
    print(f"  Train {data['X_train'].shape} pos={int(data['y_train'].sum())} | "
          f"Val {data['X_val'].shape} pos={int(data['y_val'].sum())} | "
          f"Test {data['X_test'].shape} pos={int(data['y_test'].sum())}")

    y_va, y_te = data["y_val"], data["y_test"]
    val_probs_all, test_probs_all = [], []

    for s in range(args.seeds):
        seed = 42 + s
        model = train_seed(seed, data, args, device)
        val_probs_all.append(predict_probs(model, data["X_val"], device))
        test_probs_all.append(predict_probs(model, data["X_test"], device))

    # Deep-ensemble = mean of per-seed probabilities.
    val_ens = np.mean(val_probs_all, axis=0)
    test_ens = np.mean(test_probs_all, axis=0)

    # Isotonic calibration on validation, then F1-optimal threshold on validation.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_ens, y_va)
    val_cal = iso.predict(val_ens)
    test_cal = iso.predict(test_ens)
    thr = best_f1_threshold(y_va, val_cal)

    res = {
        "seeds": args.seeds,
        "d_model": args.d_model,
        "loss": args.loss,
        "tuned_threshold": float(thr),
        "test_default_0.5": metrics_at(y_te, test_ens, 0.5),
        "test_calibrated_tuned": metrics_at(y_te, test_cal, thr),
        "test_single_seed_0.5": metrics_at(y_te, test_probs_all[0], 0.5),
    }

    d05, dtu = res["test_default_0.5"], res["test_calibrated_tuned"]
    print("\n" + "=" * 78)
    print("  GTAN deep-ensemble — TEST results")
    print("=" * 78)
    print(f"  @0.5 (ensemble raw):     acc={d05['accuracy']:.4f}  "
          f"prec={d05['precision']:.4f}  rec={d05['recall']:.4f}  "
          f"F1={d05['f1']:.4f}  AUC={d05['auc_roc']:.4f}  PR_AUC={d05['pr_auc']:.4f}")
    print(f"  @{thr:.2f} (calibrated):    acc={dtu['accuracy']:.4f}  "
          f"prec={dtu['precision']:.4f}  rec={dtu['recall']:.4f}  "
          f"F1={dtu['f1']:.4f}  AUC={dtu['auc_roc']:.4f}  PR_AUC={dtu['pr_auc']:.4f}")

    # Persist ensemble probabilities for the combined evaluator.
    PROB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROB_DIR / "gtan_val_cal.npy", val_cal)
    np.save(PROB_DIR / "gtan_test_cal.npy", test_cal)

    out = ROOT / "results_improved.json"
    payload = json.loads(out.read_text()) if out.exists() else {}
    payload["gtan"] = res
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Recorded metrics -> {out}")


if __name__ == "__main__":
    main()
