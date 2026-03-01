"""
tune.py — HPO entrypoint using Optuna + MLflow nested runs.

Usage:
    python ml/tune.py
    python ml/tune.py --trials 30
    python ml/tune.py --trials 10 --no-train

Steps:
    1. Load processed tensors from data/processed/
    2. Open parent MLflow run in experiment "nextstep-hpo"
    3. Run N Optuna trials (each = child MLflow run)
    4. Save best_params to data/processed/best_params.json
    5. Unless --no-train: retrain with best params and register @staging + @prod
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from train import load_tensors  # noqa: E402 — single source of truth
from training import MLflowRegistry, TrainConfig  # noqa: E402
from training.hpo import HPORunner  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
BEST_PARAMS_PATH = PROCESSED_DIR / "best_params.json"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
HPO_EXPERIMENT = os.getenv("MLFLOW_HPO_EXPERIMENT", "nextstep-hpo")
TRAIN_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "nextstep-lstm")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "nextstep-lstm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("tune")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter optimisation for LSTM risk model")
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials to run (default: 30)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip final retraining with best params (just save best_params.json)",
    )
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_tensors()
    input_size = X_train.shape[-1]
    log.info(
        "Loaded tensors — input_size=%d  train=%d  val=%d  test=%d", input_size, len(X_train), len(X_val), len(X_test)
    )

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        dtype=torch.float32,
    )

    # ── HPO phase ─────────────────────────────────────────────────────────────
    hpo_registry = MLflowRegistry(MLFLOW_URI, HPO_EXPERIMENT, MODEL_NAME)
    parent_run_id = hpo_registry.start_parent_run(f"hpo-{args.trials}-trials")

    try:
        runner = HPORunner(n_trials=args.trials)
        best_params = runner.run(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_size=input_size,
            pos_weight=pos_weight,
            parent_run_id=parent_run_id,
            registry=hpo_registry,
        )
    finally:
        hpo_registry.end_parent_run()

    # Save best params
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
    log.info("Best params saved to %s:\n%s", BEST_PARAMS_PATH, json.dumps(best_params, indent=2))

    # ── Optional: retrain with best params ────────────────────────────────────
    if args.no_train:
        log.info("--no-train set — skipping final training run")
        return

    log.info("Retraining with best params and registering @staging + @prod …")
    # Import train lazily to avoid circular imports at module level
    from train import train  # noqa: E402  (same ml/ dir)

    best_cfg = TrainConfig(
        hidden_size=int(best_params.get("hidden_size", 64)),
        num_layers=int(best_params.get("num_layers", 1)),
        dropout=float(best_params.get("dropout", 0.0)),
        epochs=int(best_params.get("epochs", 80)),
        lr=float(best_params.get("lr", 1e-3)),
        batch_size=int(best_params.get("batch_size", 32)),
    )
    result = train(config=best_cfg)
    log.info(
        "Final model registered — AUC=%.4f  F1=%.4f  threshold=%.4f",
        result.val_auc,
        result.val_f1,
        result.threshold,
    )


if __name__ == "__main__":
    main()
