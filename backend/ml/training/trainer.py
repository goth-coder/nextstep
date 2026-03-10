"""
TrainingLoop — single responsibility: run the PyTorch training loop.

No MLflow, no data loading, no threshold selection.
Accepts a model and tensors; returns the final train loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable  # noqa: F401 — Callable used in string annotation below

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    pass  # Callable already imported above for string annotations

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """All hyperparameters that control training. Fully serialisable."""

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    epochs: int = 80
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4  # L2 regularisation — effective for any num_layers
    pos_weight_multiplier: float = 1.0  # scales base neg/pos ratio in BCEWithLogitsLoss
    seed: int = 42
    # Logged to MLflow but not used by TrainingLoop directly
    extra_meta: dict = field(default_factory=dict)

    def to_mlflow_params(self) -> dict[str, str]:
        return {
            "hidden_size": str(self.hidden_size),
            "num_layers": str(self.num_layers),
            "dropout": str(self.dropout),
            "epochs": str(self.epochs),
            "lr": str(self.lr),
            "batch_size": str(self.batch_size),
            "weight_decay": str(self.weight_decay),
            "pos_weight_multiplier": str(self.pos_weight_multiplier),
            "seed": str(self.seed),
            **{k: str(v) for k, v in self.extra_meta.items()},
        }


class TrainingLoop:
    """
    Encapsulates the PyTorch training loop.

    SRP: knows nothing about MLflow, data files, or evaluation.
    """

    def __init__(self, config: TrainConfig, pos_weight: torch.Tensor) -> None:
        self._cfg = config
        # pos_weight (neg/pos ratio) is retained for API compat but is no longer
        # applied directly.  WeightedRandomSampler already balances each batch to
        # ~50/50, so pos_weight_multiplier in the loss is used instead (see fit()).
        self._pos_weight = pos_weight  # noqa: F841 — kept for API compatibility

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        *,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        patience: int = 15,
        step_callback: "Callable[[int, float], None] | None" = None,
    ) -> list[float]:
        """
        Train `model` in-place. Returns per-epoch average loss curve.

        Args:
            model: initialised LSTMClassifier (will be mutated).
            X_train: (N, 1, input_size) float32 tensor.
            y_train: (N,) float32 binary labels.
            X_val / y_val: optional validation tensors for early stopping.
                When provided, training stops when val F1 has not improved
                for `patience` consecutive epochs (best model weights restored).
            patience: early-stopping patience in epochs (default 15).
            step_callback: optional fn(epoch, loss) for HPO pruning hooks.
        """
        torch.manual_seed(self._cfg.seed)
        # pos_weight: effective class weight = base neg/pos ratio × multiplier.
        # The model sees the TRUE 17% positive rate in every mini-batch (no
        # artificial resampling), so BCEWithLogitsLoss(pos_weight=...) is the
        # sole lever for class-imbalance compensation.
        # pos_weight_multiplier then adjusts the precision/recall tradeoff:
        #   < 1.0  →  precision-biased (fewer false alarms)
        #   = 1.0  →  full theoretical compensation for the neg/pos ratio
        #   > 1.0  →  recall-biased (fewer missed positives)
        # NOT using WeightedRandomSampler: it was forcing 50/50 batches, which
        # caused logit saturation (train_loss → 0, bimodal score distribution).
        effective_pw = self._pos_weight * self._cfg.pos_weight_multiplier
        criterion = nn.BCEWithLogitsLoss(pos_weight=effective_pw)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._cfg.lr, weight_decay=self._cfg.weight_decay)

        # Early stopping state
        _use_early_stop = X_val is not None and y_val is not None
        _best_val_f1 = -1.0
        _best_state: dict | None = None
        _patience_left = patience

        ds = TensorDataset(X_train, y_train)
        loader = DataLoader(ds, batch_size=self._cfg.batch_size, shuffle=True)

        loss_curve: list[float] = []
        model.train()
        n = len(X_train)

        for epoch in range(1, self._cfg.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                # Gradient clipping: prevents logit saturation that causes train_loss→0
                # and bimodal score distribution. Max norm of 1.0 is conservative.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            avg = epoch_loss / n
            loss_curve.append(avg)

            if step_callback:
                step_callback(epoch, avg)

            # Early stopping: monitor val F1 every 5 epochs to reduce overhead
            if _use_early_stop and epoch % 5 == 0:
                from sklearn.metrics import f1_score
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val).squeeze(-1).numpy()
                # Use sigmoid probs directly (no calibrator during training loop)
                val_probs = 1.0 / (1.0 + np.exp(-val_logits))
                # Dynamic threshold: midpoint of prob range
                thresh = float(val_probs.mean())
                val_preds = (val_probs >= thresh).astype(int)
                val_f1 = float(f1_score(y_val.numpy().astype(int), val_preds, zero_division=0))
                if val_f1 > _best_val_f1:
                    _best_val_f1 = val_f1
                    import copy
                    _best_state = copy.deepcopy(model.state_dict())
                    _patience_left = patience
                else:
                    _patience_left -= 1
                    if _patience_left <= 0:
                        log.info(
                            "Early stopping at epoch %d — best val_f1=%.4f  train_loss=%.4f",
                            epoch, _best_val_f1, avg,
                        )
                        break

            if epoch % 10 == 0 or epoch == self._cfg.epochs:
                log.info("Epoch %d/%d — train_loss=%.4f", epoch, self._cfg.epochs, avg)

        # Restore best weights if early stopping was active
        if _use_early_stop and _best_state is not None:
            model.load_state_dict(_best_state)
            log.info("Restored best model weights (val_f1=%.4f)", _best_val_f1)

        return loss_curve
