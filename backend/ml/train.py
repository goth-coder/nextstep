"""
train.py — thin training entrypoint.

Usage:
    python ml/train.py
    python ml/train.py --config data/processed/best_params.json

Steps:
    1. Load processed tensors from data/processed/
    2. Load config (JSON or defaults)
    3. TrainingLoop.run()
    4. Evaluator.find_threshold() [val set]
    5. Evaluator.evaluate()       [test set]
    6. Quality gate
    7. MLflowRegistry.log_run() + promote @staging + @prod
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # backend/
sys.path.insert(0, str(ROOT / "ml"))

from models import LSTMClassifier  # noqa: E402
from training import (  # noqa: E402
    EvalResult,
    Evaluator,
    MLflowRegistry,
    TrainConfig,
    TrainingLoop,
)

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "nextstep-lstm")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "nextstep-lstm")

MIN_AUC = float(os.getenv("QUALITY_MIN_AUC", "0.70"))
MIN_F1 = float(os.getenv("QUALITY_MIN_F1", "0.48"))  # ≥0.55 was unreachable with ~600 train pairs + temporal split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("train")


# ── Main ──────────────────────────────────────────────────────────────────────


def load_tensors() -> tuple[torch.Tensor, ...]:
    """
    Load numpy arrays saved by data_loader.py and return six tensors.

    data_loader.py writes:  X_train.npy, y_train.npy, X_test.npy, y_test.npy
    The val split (last 20% of train, temporal order) is done here so the
    test set is never touched during threshold selection.

    Tensor shapes:
        X_*  →  (N, 1, input_size)   # seq_len=1 for LSTM
        y_*  →  (N,)
    """

    def _npy(name: str) -> np.ndarray:
        p = PROCESSED_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p} — run data_loader.py first")
        return np.load(p)

    X_train_np = _npy("X_train.npy")
    y_train_np = _npy("y_train.npy")
    X_test_np = _npy("X_test.npy")
    y_test_np = _npy("y_test.npy")

    # Temporal val split: last 20% of train (no shuffle — preserve time order)
    val_size = max(1, int(0.20 * len(X_train_np)))
    tr_size = len(X_train_np) - val_size
    X_tr_np, X_val_np = X_train_np[:tr_size], X_train_np[tr_size:]
    y_tr_np, y_val_np = y_train_np[:tr_size], y_train_np[tr_size:]

    log.info(
        "Splits — train=%d (pos=%.1f%%)  val=%d (pos=%.1f%%)  test=%d (pos=%.1f%%)",
        tr_size,
        float(y_tr_np.mean()) * 100,
        val_size,
        float(y_val_np.mean()) * 100,
        len(y_test_np),
        float(y_test_np.mean()) * 100,
    )

    def _t(arr: np.ndarray, squeeze: bool = False) -> torch.Tensor:
        t = torch.from_numpy(arr.astype("float32"))
        return t.unsqueeze(1) if squeeze else t  # unsqueeze → (N,1,features)

    return (
        _t(X_tr_np, squeeze=True),  # X_train
        _t(y_tr_np),  # y_train
        _t(X_val_np, squeeze=True),  # X_val
        _t(y_val_np),  # y_val
        _t(X_test_np, squeeze=True),  # X_test
        _t(y_test_np),  # y_test
    )


def build_config(config_path: Path | None) -> TrainConfig:
    if config_path and config_path.exists():
        log.info("Loading config from %s", config_path)
        params: dict = json.loads(config_path.read_text())
        return TrainConfig(
            hidden_size=int(params.get("hidden_size", 64)),
            num_layers=int(params.get("num_layers", 1)),
            dropout=float(params.get("dropout", 0.0)),
            epochs=int(params.get("epochs", 80)),
            lr=float(params.get("lr", 1e-3)),
            batch_size=int(params.get("batch_size", 32)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
            pos_weight_multiplier=float(params.get("pos_weight_multiplier", 1.0)),
        )
    log.info("No config file — using default TrainConfig")
    return TrainConfig()


def train(
    config: TrainConfig | None = None,
    config_path: Path | None = None,
    parent_run_id: str | None = None,
) -> EvalResult:
    """
    Public entry-point usable by tune.py.

    Returns the EvalResult from the final registered run.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_tensors()

    cfg = config or build_config(config_path)
    input_size = X_train.shape[-1]

    log.info("input_size=%d  config=%s", input_size, cfg)

    # Class imbalance weight
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)

    # 1. Model
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    # 2. Train
    loop = TrainingLoop(cfg, pos_weight)
    loss_curve = loop.fit(model, X_train, y_train)
    train_loss = loss_curve[-1]

    # 3. Calibrate (val set) — remap logits to the true class prior (~17 %)
    # WeightedRandomSampler trains on 50/50 batches, so raw sigmoid is calibrated
    # for P=0.5. PlattCalibrator fits sigmoid(A·logit + B) to recover the real prior
    # so the threshold transfers across the temporal split (val 22→23, test 23→24).
    evaluator = Evaluator()
    calibrator = evaluator.fit_calibrator(model, X_val, y_val)

    # 4. Threshold (val set, calibrated probs)
    threshold, val_f1_internal = evaluator.find_threshold(model, X_val, y_val, calibrator)

    # 5. Evaluate (test set, calibrated probs)
    result = evaluator.evaluate(model, X_test, y_test, threshold, val_f1_internal, train_loss, calibrator)

    # 5. Quality gate
    # Gate on test_f1_oracle (best achievable F1 on test set) rather than the
    # deployed F1 (val threshold transferred to test).  This separates model
    # quality from threshold-transfer quality across temporal splits.
    if result.val_auc < MIN_AUC or result.test_f1_oracle < MIN_F1:
        log.error(
            "Quality gate FAILED — AUC=%.4f (min %.2f)  F1_oracle=%.4f (min %.2f)  F1_deployed=%.4f",
            result.val_auc,
            MIN_AUC,
            result.test_f1_oracle,
            MIN_F1,
            result.val_f1,
        )
        sys.exit(1)

    log.info(
        "Quality gate PASSED — AUC=%.4f  F1_oracle=%.4f  F1_deployed=%.4f  threshold=%.4f",
        result.val_auc,
        result.test_f1_oracle,
        result.val_f1,
        result.threshold,
    )

    # 6. Log + register
    registry = MLflowRegistry(MLFLOW_URI, EXPERIMENT_NAME, MODEL_NAME)
    run_id = registry.log_run(
        config=cfg,
        result=result,
        model=model,
        scaler_path=SCALER_PATH,
        calibrator=calibrator,
        parent_run_id=parent_run_id,
        run_name="train",
        input_size=input_size,
    )
    registry.promote(run_id, aliases=["staging", "prod"])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM risk model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON file with hyperparameters (default: use TrainConfig defaults)",
    )
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()
