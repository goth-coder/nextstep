"""
MLflow model repository — concrete implementation of ModelRepository port.

Loads either the LSTM (`nextstep-lstm`) or LightGBM (`nextstep-lgbm`) model
registered under the `@prod` alias from MLflow, dispatching on the `model_type`
tag set at training time.  The rest of the application never imports torch,
mlflow, or lightgbm directly.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

MODEL_ALIAS = "prod"


class MLflowModelRepository:
    """Implements ModelRepository port using MLflow. Supports LSTM and LGBM."""

    def __init__(
        self,
        model_name: str = "nextstep-lstm",
        alias: str = MODEL_ALIAS,
        tracking_uri: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._alias = alias
        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self._model = None          # torch.nn.Module | lgb.LGBMClassifier
        self._model_type = "lstm"   # detected from MLflow tag
        self._calibrator: object | None = None

    # ── ModelRepository protocol ──────────────────────────────────────────────

    def load(self) -> None:
        import sys
        from pathlib import Path

        import mlflow
        import mlflow.lightgbm
        import mlflow.pytorch

        mlflow.set_tracking_uri(self._tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=self._tracking_uri)
        version = client.get_model_version_by_alias(self._model_name, self._alias)
        run_id = version.run_id

        # Detect model type from training tag (fallback: assume lstm)
        run = client.get_run(run_id)
        self._model_type = run.data.tags.get("model_type", "lstm")
        log.info("Loading model_type=%s  name=%s  alias=%s", self._model_type, self._model_name, self._alias)

        uri = f"models:/{self._model_name}@{self._alias}"
        if self._model_type == "lgbm":
            self._model = mlflow.lightgbm.load_model(uri)
            log.info("LightGBM model loaded ✓")
        else:
            # Add ml/ to path so MLflow can unpickle LSTMClassifier
            src_path = str(Path(__file__).parent.parent.parent / "ml")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            self._model = mlflow.pytorch.load_model(uri)
            self._model.eval()
            log.info("LSTM model loaded ✓")

        # Load Platt calibrator (present on all models trained after calibration PR)
        import pickle
        try:
            local_path = client.download_artifacts(run_id, "calibrator/calibrator.pkl")
            with open(local_path, "rb") as _f:
                self._calibrator = pickle.load(_f)
            log.info("Calibrator loaded ✓ (Platt scaling active)")
        except Exception as _exc:
            log.warning("Calibrator not found (%s) — falling back to raw probs", _exc)
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

        if self._model_type == "lgbm":
            return self._predict_lgbm(X)
        return self._predict_lstm(X)

    # ── Private dispatch ──────────────────────────────────────────────────────

    def _predict_lstm(self, X: np.ndarray) -> np.ndarray:
        import torch
        tensor = torch.from_numpy(X.astype("float32")).unsqueeze(1)
        with torch.no_grad():
            logits = self._model(tensor).squeeze(-1).numpy().flatten()
        if self._calibrator is not None:
            return self._calibrator.predict_proba(logits).astype("float32")
        return (1.0 / (1.0 + np.exp(-logits))).astype("float32")

    def _predict_lgbm(self, X: np.ndarray) -> np.ndarray:
        import numpy as np
        probs = self._model.predict_proba(X.astype("float32"))[:, 1]
        if self._calibrator is not None:
            probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
            logits = np.log(probs_clipped / (1.0 - probs_clipped))
            return self._calibrator.predict_proba(logits).astype("float32")
        return probs.astype("float32")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
