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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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
        step_callback: "Callable[[int, float], None] | None" = None,
    ) -> list[float]:
        """
        Train `model` in-place. Returns per-epoch average loss curve.

        Args:
            model: initialised LSTMClassifier (will be mutated).
            X_train: (N, 1, input_size) float32 tensor.
            y_train: (N,) float32 binary labels.
            step_callback: optional fn(epoch, loss) for HPO pruning hooks.
        """
        torch.manual_seed(self._cfg.seed)
        # WeightedRandomSampler (below) already makes every batch ~50/50
        # positive/negative.  Applying the raw neg/pos ratio (~4.8×) on top via
        # BCEWithLogitsLoss(pos_weight=...) would double-compensate for the
        # imbalance, pushing the model toward extremely high recall at the cost
        # of precision and collapsing F1.
        #
        # With balanced batches the "neutral" pos_weight is 1.0.
        # pos_weight_multiplier then becomes the sole precision/recall dial:
        #   < 1.0  →  precision-biased
        #   = 1.0  →  balanced (neutral, correct for balanced sampler)
        #   > 1.0  →  recall-biased
        # Optuna searches this in [0.5, 4.0], giving it full freedom.
        scaled_pw = torch.tensor([self._cfg.pos_weight_multiplier])
        criterion = nn.BCEWithLogitsLoss(pos_weight=scaled_pw)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._cfg.lr, weight_decay=self._cfg.weight_decay)

        ds = TensorDataset(X_train, y_train)
        # WeightedRandomSampler: ensure each batch has a balanced view of the
        # minority class.  With ~17% positives and batch_size≤64, random
        # shuffling can produce all-negative batches → wasted gradient steps.
        labels_np = y_train.numpy()
        n_pos = int(labels_np.sum())
        n_neg = len(labels_np) - n_pos
        sample_weights = torch.where(
            y_train == 1,
            torch.tensor(1.0 / max(n_pos, 1)),
            torch.tensor(1.0 / max(n_neg, 1)),
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader = DataLoader(ds, batch_size=self._cfg.batch_size, sampler=sampler)

        loss_curve: list[float] = []
        model.train()
        n = len(X_train)

        for epoch in range(1, self._cfg.epochs + 1):
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            avg = epoch_loss / n
            loss_curve.append(avg)

            if step_callback:
                step_callback(epoch, avg)

            if epoch % 10 == 0 or epoch == self._cfg.epochs:
                log.info("Epoch %d/%d — train_loss=%.4f", epoch, self._cfg.epochs, avg)

        return loss_curve
