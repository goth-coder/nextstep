"""
LSTM training pipeline: loads processed tensors, trains the model, logs to MLflow,
registers the model, and promotes it to the @staging alias.

Usage:
    python src/train_lstm.py
    MLFLOW_TRACKING_URI=http://localhost:5000 python src/train_lstm.py
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train_lstm")

# ── Constants / Hyperparameters ───────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent.parent
PROCESSED_DIR = _BACKEND_DIR / "data" / "processed"

MODEL_NAME = "nextstep-lstm"
INPUT_SIZE = 16  # matches data_loader.py FEATURES list
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 32
SEED = 42

MIN_AUC = 0.70
MIN_F1 = 0.55


# ── Model Definition ──────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """Single-layer LSTM → Linear(hidden_size, 1). Outputs raw logits."""

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len=1, input_size=9)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(1)  # raw logits (batch,)


# ── Training ──────────────────────────────────────────────────────────────────
def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Load pre-split tensors (temporal split done in data_loader.py) ─────
    files = {
        "X_train": PROCESSED_DIR / "X_train.npy",
        "y_train": PROCESSED_DIR / "y_train.npy",
        "X_test": PROCESSED_DIR / "X_test.npy",
        "y_test": PROCESSED_DIR / "y_test.npy",
        "scaler": PROCESSED_DIR / "scaler.pkl",
    }
    for name, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p} — run data_loader.py first")

    X_train_np = np.load(files["X_train"])  # (N_train, 9)
    y_train_np = np.load(files["y_train"])  # (N_train,)
    X_test_np = np.load(files["X_test"])  # (N_test, 9)
    y_test_np = np.load(files["y_test"])  # (N_test,)

    n_train, n_test = len(X_train_np), len(X_test_np)
    log.info(
        "Train: %d rows (pos=%.1f%%)  |  Test: %d rows (pos=%.1f%%)",
        n_train,
        float(y_train_np.mean()) * 100,
        n_test,
        float(y_test_np.mean()) * 100,
    )

    # ── 2. Temporal validation split from train (last 20%) ──────────────────
    # We split the train set keeping temporal order (no shuffle).
    # Validation is used ONLY to find the optimal classification threshold
    # via the PR curve — it is never used for model weight updates.
    # This ensures the test set is never touched during threshold selection.
    val_size = max(1, int(0.20 * n_train))
    tr_size = n_train - val_size
    X_tr_np, X_val_np = X_train_np[:tr_size], X_train_np[tr_size:]
    y_tr_np, y_val_np = y_train_np[:tr_size], y_train_np[tr_size:]
    log.info(
        "Val split: tr=%d (pos=%.1f%%)  val=%d (pos=%.1f%%)",
        tr_size,
        float(y_tr_np.mean()) * 100,
        val_size,
        float(y_val_np.mean()) * 100,
    )

    # ── 3. Reshape for LSTM: (N, seq_len=1, input_size) ────────────────────
    X_tr_t = torch.from_numpy(X_tr_np).unsqueeze(1)
    y_tr_t = torch.from_numpy(y_tr_np)
    X_val_t = torch.from_numpy(X_val_np).unsqueeze(1)
    y_val_t = torch.from_numpy(y_val_np)
    X_test_t = torch.from_numpy(X_test_np).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test_np)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Compute pos_weight for class imbalance: neg/pos ratio (from full train, not just tr split)
    n_pos = float(y_train_np.sum())
    n_neg = n_train - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    log.info("pos_weight for BCEWithLogitsLoss: %.4f", pos_weight.item())

    # ── 4. Configure MLflow ───────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("nextstep-lstm-training")
    log.info("MLflow tracking URI: %s", tracking_uri)

    # ── 5. Train with MLflow run ──────────────────────────────────────────────
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info("MLflow run started: run_id=%s", run_id)

        # Log hyperparameters
        mlflow.log_params(
            {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "epochs": EPOCHS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "split": "temporal_22-23_train__23-24_test",
                "val_split": "last_20pct_of_train_temporal",
                "input_size": INPUT_SIZE,
                "seed": SEED,
                "scaler": "RobustScaler",
                "threshold_method": "PR_curve_F1_max_on_val",
                "extra_features": "gender+age",
            }
        )

        model = LSTMClassifier(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Training loop
        model.train()
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)  # raw logits
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            avg_loss = epoch_loss / tr_size
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if epoch % 10 == 0 or epoch == EPOCHS:
                log.info("Epoch %d/%d — train_loss=%.4f", epoch, EPOCHS, avg_loss)

        # ── 6a. Find optimal threshold on VALIDATION set (PR curve) ────────────
        # Threshold is tuned on val (20% held-out from train), NEVER on test.
        # This avoids the subtle leakage of using test-set distribution to pick threshold.
        model.eval()
        val_probs_list: list[float] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                probs = torch.sigmoid(model(X_batch))
                val_probs_list.extend(probs.numpy().tolist())

        val_probs_np = np.array(val_probs_list)
        if np.isnan(val_probs_np).any():
            val_probs_np = np.nan_to_num(val_probs_np, nan=0.5)

        # Sweep thresholds to maximise F1 on validation set
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_val_np.astype(int), val_probs_np)
        f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
        optimal_idx = int(np.argmax(f1_arr))
        threshold = float(thr_arr[optimal_idx])
        val_f1_internal = float(f1_arr[optimal_idx])
        log.info(
            "Optimal threshold from val PR curve: %.4f  (val F1=%.4f)",
            threshold,
            val_f1_internal,
        )
        mlflow.log_metrics({"threshold": threshold, "val_f1_internal": val_f1_internal})

        # ── 6b. Evaluate on TEST set with val-derived threshold ───────────────
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.numpy().tolist())
                all_labels.extend(y_batch.numpy().tolist())

        all_probs_np = np.array(all_probs)
        all_labels_np = np.array(all_labels, dtype=int)

        if np.isnan(all_probs_np).any():
            nan_count = int(np.isnan(all_probs_np).sum())
            log.warning("NaN in %d/%d test predictions — replacing with 0.5", nan_count, len(all_probs_np))
            all_probs_np = np.nan_to_num(all_probs_np, nan=0.5)

        all_preds = (all_probs_np >= threshold).astype(int)
        val_auc = roc_auc_score(all_labels_np, all_probs_np)
        val_f1 = f1_score(all_labels_np, all_preds, zero_division=0)

        mlflow.log_metrics({"val_auc": val_auc, "val_f1": val_f1})
        log.info("Test evaluation — AUC=%.4f  F1=%.4f  (threshold=%.4f from val)", val_auc, val_f1, threshold)

        # ── 7. Quality gate before registration ───────────────────────────────
        if val_auc < MIN_AUC or val_f1 < MIN_F1:
            raise ValueError(
                f"Model failed quality gate: val_auc={val_auc:.4f} (min {MIN_AUC}), "
                f"val_f1={val_f1:.4f} (min {MIN_F1}). "
                "Increase EPOCHS or adjust architecture before registering."
            )

        log.info("Quality gate passed ✓  AUC=%.4f  F1=%.4f", val_auc, val_f1)

        # ── 8. Log scaler artifact ────────────────────────────────────────────
        mlflow.log_artifact(str(files["scaler"]), artifact_path="scaler")

        # ── 9. Register model ─────────────────────────────────────────────────
        log.info("Registering model '%s' in MLflow Model Registry…", MODEL_NAME)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

    # ── 10. Promote to @staging and @prod aliases ────────────────────────────
    client = MlflowClient(tracking_uri=tracking_uri)
    # Get the latest version just created
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, "staging", str(latest_version))
    client.set_registered_model_alias(MODEL_NAME, "prod", str(latest_version))
    log.info("Model v%d promoted to @staging and @prod aliases ✓", latest_version)

    log.info("Training pipeline complete ✓")


if __name__ == "__main__":
    train()
