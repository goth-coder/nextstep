"""
TrainingLoop — single responsibility: run the PyTorch training loop.

No MLflow, no data loading, no threshold selection.
Accepts a model and tensors; returns the final train loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
        self._pos_weight = pos_weight

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
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._cfg.lr)

        ds = TensorDataset(X_train, y_train)
        loader = DataLoader(ds, batch_size=self._cfg.batch_size, shuffle=True)

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
