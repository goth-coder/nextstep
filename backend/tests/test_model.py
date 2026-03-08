"""
Model performance gate test.

Loads the trained LSTM from MLflow @prod alias and asserts:
  - val_auc >= 0.70
  - val_f1  >= 0.55

Uses the pre-split X_test.npy / y_test.npy produced by data_loader.py
(temporal split: 2023 features → 2024 labels).

The optimal threshold is read from the MLflow run that produced @prod.
Falls back to positive_rate if the metric is not logged.

Usage:
    pytest backend/tests/test_model.py -v

Requires:
    - MLflow server running (MLFLOW_TRACKING_URI set or local mlruns/)
    - Model registered with @prod alias
    - Processed tensors in backend/data/processed/
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score, roc_auc_score

MIN_AUC = 0.70  # must match train_lstm.py
MIN_F1 = 0.55  # must match train_lstm.py

_BACKEND_DIR = Path(__file__).parent.parent
PROCESSED_DIR = _BACKEND_DIR / "data" / "processed"

_TENSOR_SKIP = pytest.mark.skipif(
    not (PROCESSED_DIR / "X_test.npy").exists(),
    reason="X_test.npy not found — run data_loader.py first",
)
# Skip when no remote MLflow is configured. A local mlruns/ folder may exist but
# contain incomplete artifacts from past runs — only treat a remote URI as available.
_MLFLOW_SKIP = pytest.mark.skipif(
    os.getenv("MLFLOW_TRACKING_URI") is None,
    reason="MLFLOW_TRACKING_URI not set — skipping model gate (CI sets this env var)",
)


def _load_prod_model_and_threshold():
    import mlflow
    import mlflow.pytorch
    from mlflow import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model = mlflow.pytorch.load_model("models:/nextstep-lstm@prod")
    model.eval()

    # Try to retrieve the threshold logged during training
    client = MlflowClient(tracking_uri)
    alias_mv = client.get_model_version_by_alias("nextstep-lstm", "prod")
    run = client.get_run(alias_mv.run_id)
    threshold = float(run.data.metrics.get("threshold", 0.5))
    return model, threshold


@_TENSOR_SKIP
@_MLFLOW_SKIP
def test_model_performance_gate():
    """Assert that the @prod model meets minimum AUC and F1 thresholds
    on the temporal test split (2023→2024)."""
    model, threshold = _load_prod_model_and_threshold()

    # Load the temporal test split produced by ETL
    X_test_np = np.load(PROCESSED_DIR / "X_test.npy")  # (N_test, 8)
    y_test_np = np.load(PROCESSED_DIR / "y_test.npy")  # (N_test,)

    X_test_t = torch.from_numpy(X_test_np).unsqueeze(1).float()  # (N, 1, 8)

    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).numpy().flatten()  # model outputs raw logits

    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_test_np.astype(int), probs)
    f1 = f1_score(y_test_np.astype(int), preds, zero_division=0)

    print(f"\n[Model Gate] AUC={auc:.4f}  F1={f1:.4f}  threshold={threshold:.4f}")

    assert auc >= MIN_AUC, f"AUC {auc:.4f} below threshold {MIN_AUC}"
    assert f1 >= MIN_F1, f"F1 {f1:.4f} below threshold {MIN_F1}"
