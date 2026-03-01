"""
HPORunner — Optuna-based hyperparameter optimisation.

SRP: build the search space, run N trials (each a child MLflow run),
     return the best hyperparameter dict.

No MLflow parent-run management, no data loading, no registration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import optuna
import torch

from models import LSTMClassifier
from .evaluator import Evaluator
from .trainer import TrainConfig, TrainingLoop

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Suppress noisy Optuna default logs unless DEBUG is set
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HPORunner:
    """
    Runs `n_trials` Optuna trials.

    Each trial:
    1. Samples hyperparameters from the search space.
    2. Trains a model using TrainingLoop.
    3. Evaluates on the val set only (objective: val_f1_internal).
    4. Logs the trial as a child MLflow run under `parent_run_id`.

    Returns the best hyperparameter dict (compatible with TrainConfig).
    """

    # Search space definition
    _HIDDEN_SIZES = [32, 64, 128, 256]
    _BATCH_SIZES = [16, 32, 64]
    _NUM_LAYERS = [1, 2]

    def __init__(self, n_trials: int = 30) -> None:
        self._n_trials = n_trials
        self._evaluator = Evaluator()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        input_size: int,
        pos_weight: torch.Tensor,
        parent_run_id: str,
        registry: "MLflowRegistry",  # noqa: F821 — resolved at runtime
    ) -> dict:
        """
        Execute HPO and return the best hyperparameter dict.

        Args:
            X_train / y_train: training split tensors.
            X_val   / y_val  : validation split tensors (used for objective).
            input_size       : number of LSTM input features.
            pos_weight       : class-imbalance weight tensor.
            parent_run_id    : MLflow parent run ID for child run nesting.
            registry         : MLflowRegistry instance for logging child runs.
        """

        def objective(trial: optuna.Trial) -> float:
            cfg = self._sample_config(trial)
            model = LSTMClassifier(
                input_size=input_size,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
            )
            loop = TrainingLoop(cfg, pos_weight)
            loss_curve = loop.fit(model, X_train, y_train)
            train_loss = loss_curve[-1]

            threshold, val_f1 = self._evaluator.find_threshold(model, X_val, y_val)

            # Log child run to MLflow
            from .evaluator import EvalResult

            result = EvalResult(
                threshold=threshold,
                val_f1_internal=val_f1,
                val_auc=0.0,  # not computed during HPO (test set unseen)
                val_f1=0.0,  # idem
                train_loss=train_loss,
            )
            registry.log_run(
                config=cfg,
                result=result,
                model=model,
                parent_run_id=parent_run_id,
                run_name=f"trial-{trial.number}",
                input_size=input_size,
            )

            log.info(
                "Trial %d — val_f1=%.4f  threshold=%.4f  params=%s",
                trial.number,
                val_f1,
                threshold,
                {k: v for k, v in cfg.to_mlflow_params().items() if k != "seed"},
            )
            return val_f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)

        best = study.best_trial
        log.info(
            "HPO finished — best trial %d  val_f1=%.4f",
            best.number,
            best.value,
        )
        return self._trial_to_dict(best)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sample_config(self, trial: optuna.Trial) -> TrainConfig:
        hidden_size = trial.suggest_categorical("hidden_size", self._HIDDEN_SIZES)
        num_layers = trial.suggest_categorical("num_layers", self._NUM_LAYERS)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        epochs = trial.suggest_int("epochs", 60, 120)
        batch_size = trial.suggest_categorical("batch_size", self._BATCH_SIZES)

        # Optuna doesn't allow dropout > 0 when num_layers == 1 in PyTorch,
        # so we clamp it here for consistency with LSTMClassifier behaviour.
        if num_layers == 1:
            dropout = 0.0

        return TrainConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )

    @staticmethod
    def _trial_to_dict(trial: optuna.trial.FrozenTrial) -> dict:
        """Convert a frozen trial's params back to a TrainConfig-compatible dict."""
        params = dict(trial.params)
        # Ensure dropout is 0.0 for single-layer models
        if params.get("num_layers", 1) == 1:
            params["dropout"] = 0.0
        return params
