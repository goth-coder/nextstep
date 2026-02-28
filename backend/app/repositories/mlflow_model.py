"""
MLflow model repository — concrete implementation of ModelRepository port.

Loads the LSTM registered under `nextstep-lstm@prod` from MLflow and exposes
a numpy-in / numpy-out predict() interface. The rest of the application never
imports torch or mlflow directly.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)

MODEL_NAME = "nextstep-lstm"
MODEL_ALIAS = "prod"


class MLflowModelRepository:
    """Implements ModelRepository port using MLflow's PyTorch flavour."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        alias: str = MODEL_ALIAS,
        tracking_uri: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._alias = alias
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._model: torch.nn.Module | None = None

    # ── ModelRepository protocol ──────────────────────────────────────────────

    def load(self) -> None:
        import mlflow.pytorch

        mlflow.set_tracking_uri(self._tracking_uri)
        uri = f"models:/{self._model_name}@{self._alias}"
        log.info("Loading model from MLflow: %s", uri)
        self._model = mlflow.pytorch.load_model(uri)
        self._model.eval()
        log.info("Model loaded ✓")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (N, input_size) float32 array — already scaled.

        Returns
        -------
        (N,) float32 — risk scores in [0, 1].
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # LSTM expects (batch, seq_len=1, features)
        tensor = torch.from_numpy(X.astype("float32")).unsqueeze(1)
        with torch.no_grad():
            raw = self._model(tensor)
            # Model may output logits (LSTMLogits) or probabilities (LSTMClassifier)
            scores = torch.sigmoid(raw) if raw.max() > 1 or raw.min() < 0 else raw
        return scores.numpy().flatten()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
