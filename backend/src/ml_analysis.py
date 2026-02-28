"""
ML Analysis Script — Sprint 1
Runs experiments E1–E5, prints a Markdown-ready summary table.

E1: Baseline (random split, full-dataset scaler) — as deployed today
E2: Temporal split (train=2021, test=2022) — expose leakage
E3: Scaler fit only on train — fix normalizer leakage
E4: E3 + Dropout(0.3) — regularization
E5: E3 + pos_weight in BCEWithLogitsLoss — improve minority-class recall

Usage (inside the api container or any env with deps):
    python src/ml_analysis.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Dataset path ───────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).parent
_BACKEND_DIR = _SRC_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent

CANDIDATES = [
    Path("/data/dataset_unificado_defasagem.csv"),
    _PROJECT_ROOT / "data" / "dataset_unificado_defasagem.csv",
    _BACKEND_DIR / "data" / "raw" / "dataset_unificado_defasagem.csv",
]

INDICATORS = ["IAA", "IEG", "IPS", "IDA", "IAN", "IPV"]
TARGET = "defasagem_bin"

HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 32


# ── Model definitions ──────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """Baseline: LSTM → Linear → Sigmoid (BCELoss)."""

    def __init__(self, input_size: int = 6, hidden_size: int = HIDDEN_SIZE, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=NUM_LAYERS, batch_first=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        out = self.fc(self.drop(h_n[-1]))
        return self.sigmoid(out).squeeze(1)


class LSTMLogits(nn.Module):
    """E5: LSTM → Linear (raw logits for BCEWithLogitsLoss)."""

    def __init__(self, input_size: int = 6, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)  # raw logits


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_df() -> pd.DataFrame:
    for p in CANDIDATES:
        if p.exists():
            print(f"[data] Loading from {p}")
            return pd.read_csv(p)
    raise FileNotFoundError("Dataset not found. Searched: " + str(CANDIDATES))


def make_tensors(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    Xt = torch.from_numpy(X.astype("float32")).unsqueeze(1)  # (N, 1, 6)
    yt = torch.from_numpy(y.astype("float32"))
    return Xt, yt


def train_eval(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    test_y: torch.Tensor,
    model: nn.Module,
    criterion,
    use_sigmoid_for_eval: bool = True,
) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_ds = TensorDataset(train_X, train_y)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED)
    )

    loss_curve = []
    model.train()
    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
        loss_curve.append(epoch_loss / len(train_X))

    model.eval()
    with torch.no_grad():
        raw = model(test_X)
        probs = torch.sigmoid(raw) if not use_sigmoid_for_eval else raw
        probs_np = probs.numpy()

    labels_np = test_y.numpy().astype(int)
    preds_np = (probs_np >= 0.5).astype(int)

    auc = roc_auc_score(labels_np, probs_np)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    acc = accuracy_score(labels_np, preds_np)
    cm = confusion_matrix(labels_np, preds_np)
    report = classification_report(labels_np, preds_np, target_names=["No Lag", "Lag"], zero_division=0)

    return {
        "loss_curve": loss_curve,
        "val_auc": round(auc, 4),
        "val_f1": round(f1, 4),
        "val_acc": round(acc, 4),
        "final_loss": round(loss_curve[-1], 4),
        "confusion_matrix": cm.tolist(),
        "report": report,
    }


# ── Experiment runners ─────────────────────────────────────────────────────────
def e1_baseline(df: pd.DataFrame) -> dict:
    """E1: Reproduce current production behaviour exactly."""
    print("\n" + "=" * 60)
    print("E1: Baseline (random split, full-dataset scaler)")
    print("=" * 60)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[INDICATORS].to_numpy(dtype="float32"))
    y = df[TARGET].to_numpy(dtype="float32")

    N = len(X)
    n_train = int(N * 0.8)
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(N, generator=g).numpy()
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    tX, ty = make_tensors(X[train_idx], y[train_idx])
    vX, vy = make_tensors(X[test_idx], y[test_idx])

    print(f"  Train: {n_train}  Test: {N - n_train}")
    print(f"  Train positive rate: {y[train_idx].mean():.2%}   Test: {y[test_idx].mean():.2%}")

    torch.manual_seed(SEED)
    model = LSTMClassifier()
    result = train_eval(tX, ty, vX, vy, model, nn.BCELoss())
    result["train_size"] = int(n_train)
    result["test_size"] = int(N - n_train)
    result["train_pos_rate"] = round(float(y[train_idx].mean()), 4)
    result["test_pos_rate"] = round(float(y[test_idx].mean()), 4)
    print(f"  → AUC={result['val_auc']}  F1={result['val_f1']}  Acc={result['val_acc']}")
    print(result["report"])
    return result


def e2_temporal(df: pd.DataFrame) -> dict:
    """E2: Temporal split — train on 2021, test on 2022."""
    print("\n" + "=" * 60)
    print("E2: Temporal split (train=2021, test=2022)")
    print("=" * 60)

    df21 = df[df["year"] == 2021].copy()
    df22 = df[df["year"] == 2022].copy()
    print(f"  2021 rows: {len(df21)}  2022 rows: {len(df22)}")

    # Scaler fitted on ALL data (same bug as E1, isolating temporal leakage only)
    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(df[INDICATORS].to_numpy(dtype="float32"))
    df_scaled = df.copy()
    df_scaled[INDICATORS] = X_all

    X_train = df_scaled.loc[df["year"] == 2021, INDICATORS].to_numpy(dtype="float32")
    y_train = df_scaled.loc[df["year"] == 2021, TARGET].to_numpy(dtype="float32")
    X_test = df_scaled.loc[df["year"] == 2022, INDICATORS].to_numpy(dtype="float32")
    y_test = df_scaled.loc[df["year"] == 2022, TARGET].to_numpy(dtype="float32")

    print(f"  Train pos rate: {y_train.mean():.2%}   Test pos rate: {y_test.mean():.2%}")

    tX, ty = make_tensors(X_train, y_train)
    vX, vy = make_tensors(X_test, y_test)

    torch.manual_seed(SEED)
    model = LSTMClassifier()
    result = train_eval(tX, ty, vX, vy, model, nn.BCELoss())
    result["train_size"] = len(X_train)
    result["test_size"] = len(X_test)
    result["train_pos_rate"] = round(float(y_train.mean()), 4)
    result["test_pos_rate"] = round(float(y_test.mean()), 4)
    print(f"  → AUC={result['val_auc']}  F1={result['val_f1']}  Acc={result['val_acc']}")
    print(result["report"])
    return result


def e3_scaler_fix(df: pd.DataFrame) -> dict:
    """E3: Temporal split + scaler fitted only on train."""
    print("\n" + "=" * 60)
    print("E3: Temporal split + scaler fit on train only")
    print("=" * 60)

    df21 = df[df["year"] == 2021].copy()
    df22 = df[df["year"] == 2022].copy()

    X_train_raw = df21[INDICATORS].to_numpy(dtype="float32")
    y_train = df21[TARGET].to_numpy(dtype="float32")
    X_test_raw = df22[INDICATORS].to_numpy(dtype="float32")
    y_test = df22[TARGET].to_numpy(dtype="float32")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)  # transform only, no fit

    print(f"  Scaler fitted on {len(X_train)} train samples only")
    print(f"  Train pos rate: {y_train.mean():.2%}   Test pos rate: {y_test.mean():.2%}")

    tX, ty = make_tensors(X_train, y_train)
    vX, vy = make_tensors(X_test, y_test)

    torch.manual_seed(SEED)
    model = LSTMClassifier()
    result = train_eval(tX, ty, vX, vy, model, nn.BCELoss())
    result["train_size"] = len(X_train)
    result["test_size"] = len(X_test)
    result["train_pos_rate"] = round(float(y_train.mean()), 4)
    result["test_pos_rate"] = round(float(y_test.mean()), 4)
    print(f"  → AUC={result['val_auc']}  F1={result['val_f1']}  Acc={result['val_acc']}")
    print(result["report"])
    return result


def e4_dropout(df: pd.DataFrame) -> dict:
    """E4: E3 setup + Dropout(0.3) after LSTM hidden state."""
    print("\n" + "=" * 60)
    print("E4: Temporal split + scaler fix + Dropout(0.3)")
    print("=" * 60)

    df21 = df[df["year"] == 2021].copy()
    df22 = df[df["year"] == 2022].copy()

    X_train_raw = df21[INDICATORS].to_numpy(dtype="float32")
    y_train = df21[TARGET].to_numpy(dtype="float32")
    X_test_raw = df22[INDICATORS].to_numpy(dtype="float32")
    y_test = df22[TARGET].to_numpy(dtype="float32")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    tX, ty = make_tensors(X_train, y_train)
    vX, vy = make_tensors(X_test, y_test)

    torch.manual_seed(SEED)
    model = LSTMClassifier(dropout=0.3)
    result = train_eval(tX, ty, vX, vy, model, nn.BCELoss())
    result["train_size"] = len(X_train)
    result["test_size"] = len(X_test)
    result["train_pos_rate"] = round(float(y_train.mean()), 4)
    result["test_pos_rate"] = round(float(y_test.mean()), 4)
    print(f"  → AUC={result['val_auc']}  F1={result['val_f1']}  Acc={result['val_acc']}")
    print(result["report"])
    return result


def e5_pos_weight(df: pd.DataFrame) -> dict:
    """E5: E3 setup + BCEWithLogitsLoss(pos_weight) for class imbalance."""
    print("\n" + "=" * 60)
    print("E5: Temporal split + scaler fix + pos_weight")
    print("=" * 60)

    df21 = df[df["year"] == 2021].copy()
    df22 = df[df["year"] == 2022].copy()

    X_train_raw = df21[INDICATORS].to_numpy(dtype="float32")
    y_train = df21[TARGET].to_numpy(dtype="float32")
    X_test_raw = df22[INDICATORS].to_numpy(dtype="float32")
    y_test = df22[TARGET].to_numpy(dtype="float32")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # pos_weight = #negatives / #positives in train set
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"  pos_weight = {pw.item():.4f}  (neg={int(n_neg)}, pos={int(n_pos)})")

    tX, ty = make_tensors(X_train, y_train)
    vX, vy = make_tensors(X_test, y_test)

    torch.manual_seed(SEED)
    model = LSTMLogits()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    result = train_eval(tX, ty, vX, vy, model, criterion, use_sigmoid_for_eval=False)
    result["train_size"] = len(X_train)
    result["test_size"] = len(X_test)
    result["pos_weight"] = round(float(pw.item()), 4)
    result["train_pos_rate"] = round(float(y_train.mean()), 4)
    result["test_pos_rate"] = round(float(y_test.mean()), 4)
    print(f"  → AUC={result['val_auc']}  F1={result['val_f1']}  Acc={result['val_acc']}")
    print(result["report"])
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_df()
    print(f"\n[data] Loaded {len(df)} rows, columns: {df.columns.tolist()}")
    print(f"[data] Year distribution:\n{df['year'].value_counts().to_string()}")
    print(f"[data] Target distribution:\n{df[TARGET].value_counts().to_string()}")
    print(f"[data] Positive rate: {df[TARGET].mean():.2%}")

    results = {}
    results["E1"] = e1_baseline(df)
    results["E2"] = e2_temporal(df)
    results["E3"] = e3_scaler_fix(df)
    results["E4"] = e4_dropout(df)
    results["E5"] = e5_pos_weight(df)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Exp':<6} {'Description':<38} {'AUC':>6} {'F1':>6} {'Acc':>6} {'Loss':>7}"
    print(header)
    print("-" * 70)

    descs = {
        "E1": "Baseline (random split, full scaler)",
        "E2": "Temporal split, full scaler",
        "E3": "Temporal split + train-only scaler",
        "E4": "E3 + Dropout(0.3)",
        "E5": "E3 + pos_weight (BCEWithLogits)",
    }

    for k, r in results.items():
        print(
            f"{k:<6} {descs[k]:<38} {r['val_auc']:>6.4f} {r['val_f1']:>6.4f} "
            f"{r['val_acc']:>6.4f} {r['final_loss']:>7.4f}"
        )

    print("=" * 70)

    # Save JSON for report generation
    out_path = _BACKEND_DIR / "data" / "ml_analysis_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "loss_curve"} for k, v in results.items()}, f, indent=2)
    print(f"\n[results] Saved to {out_path}")


if __name__ == "__main__":
    main()
