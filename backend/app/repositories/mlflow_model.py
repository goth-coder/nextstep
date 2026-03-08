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
        self._calibrator: object | None = None  # PlattCalibrator loaded alongside model

    # ── ModelRepository protocol ──────────────────────────────────────────────

    def load(self) -> None:
        import sys
        from pathlib import Path

        import mlflow.pytorch

        # MLflow pickle-serialised the model class as `models.lstm.LSTMClassifier`.
        # That module lives under backend/ml/, which gunicorn doesn't add to sys.path.
        src_path = str(Path(__file__).parent.parent.parent / "ml")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        mlflow.set_tracking_uri(self._tracking_uri)
        uri = f"models:/{self._model_name}@{self._alias}"
        log.info("Loading model from MLflow: %s", uri)
        self._model = mlflow.pytorch.load_model(uri)
        self._model.eval()
        log.info("Model loaded ✓")

        # Load Platt calibrator from the same MLflow run (if present).
        # Absent on models trained before calibration was introduced → silent fallback.
        import pickle

        try:
            client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
            version = client.get_model_version_by_alias(self._model_name, self._alias)
            local_path = client.download_artifacts(version.run_id, "calibrator/calibrator.pkl")
            with open(local_path, "rb") as _f:
                self._calibrator = pickle.load(_f)
            log.info("Calibrator loaded ✓ (Platt scaling active)")
        except Exception as _exc:
            log.warning("Calibrator not found (%s) — falling back to raw sigmoid", _exc)
            self._calibrator = None

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
            logits = self._model(tensor).squeeze(-1).numpy().flatten()

        if self._calibrator is not None:
            # Platt-calibrated probabilities aligned to the true class prior (~17 %)
            return self._calibrator.predict_proba(logits).astype("float32")
        # Fallback for models without calibrator (pre-calibration MLflow versions)
        return (1.0 / (1.0 + np.exp(-logits))).astype("float32")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
