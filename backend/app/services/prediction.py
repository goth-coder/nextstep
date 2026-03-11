"""
PredictionService — orchestrates model loading and batch inference.

Depends only on port interfaces (ModelRepository, StudentDataRepository),
never on concrete implementations. Satisfies DIP.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.domain.ports import ModelRepository, StudentDataRepository
from app.domain.student import Indicators, StudentRecord

log = logging.getLogger(__name__)

# Feature order must match data_loader.py FEATURES list exactly.
# Keys are lowercase (API convention) mapped to ETL column names.
_API_TO_FEATURE = [
    ("iaa", "IAA"),
    ("ieg", "IEG"),
    ("ips", "IPS"),
    ("ida", "IDA"),
    ("ipv", "IPV"),
    ("ian", "IAN"),
    ("inde", "INDE"),
    ("defasagem", "defasagem"),
    ("fase_num", "fase_num"),
    ("gender", "gender"),
    ("age", "age"),
    ("mat", "mat"),
    ("por", "por"),
    ("tenure", "tenure"),
    ("n_av", "n_av"),
    ("missing_grades", "missing_grades"),
]


class PredictionService:
    """
    Single Responsibility: transform raw student metadata + features
    into a list of StudentRecord with risk scores.
    """

    def __init__(
        self,
        model_repo: ModelRepository,
        data_repo: StudentDataRepository,
    ) -> None:
        self._model_repo = model_repo
        self._data_repo = data_repo

    def run_batch_inference(self) -> list[StudentRecord]:
        """
        Load model + data, run inference, return StudentRecord list with scores.
        Raises on any loading failure (caller decides on retry / 503).
        """
        self._model_repo.load()

        metadata: list[dict[str, Any]] = self._data_repo.load_metadata()
        X: np.ndarray = self._data_repo.load_features()

        log.info("Running batch inference over %d students…", len(X))
        scores: np.ndarray = self._model_repo.predict(X)
        log.info("Inference complete ✓")

        return self._build_records(metadata, scores)

    def load_students_only(self) -> list[StudentRecord]:
        """
        Load student metadata from GCS without running the model.
        Returns StudentRecord list with risk_score=None (model not required).
        """
        metadata: list[dict[str, Any]] = self._data_repo.load_metadata()
        log.info("Loaded %d students (no model inference)", len(metadata))
        return self._build_records(metadata, scores=None)

    def _build_records(
        self,
        metadata: list[dict[str, Any]],
        scores: np.ndarray | None,
    ) -> list[StudentRecord]:
        records: list[StudentRecord] = []
        for i, meta in enumerate(metadata):
            score: float | None = float(scores[i]) if scores is not None else None
            indicators = Indicators(
                iaa=meta.get("iaa"),
                ieg=meta.get("ieg"),
                ips=meta.get("ips"),
                ida=meta.get("ida"),
                ipv=meta.get("ipv"),
                ipp=meta.get("ipp"),
                ian=meta.get("ian"),
                inde=meta.get("inde"),
                defasagem=meta.get("defasagem"),
                fase_num=meta.get("fase_num"),
                mat=meta.get("mat"),
                por=meta.get("por"),
                tenure=meta.get("tenure"),
                n_av=meta.get("n_av"),
            )
            record = StudentRecord.build(
                student_id=i,
                ra=meta.get("ra", f"RA-{i}"),
                display_name=meta.get("display_name", f"Aluno-{i}"),
                phase=meta.get("phase", "N/A"),
                phase_num=meta.get("fase_num", 0),
                class_group=meta.get("class_group", "N/A"),
                gender=int(meta.get("gender", 0)),
                age=meta.get("age"),
                year=meta.get("year", 2024),
                risk_score=score,
                indicators=indicators,
            )
            records.append(record)
        return records

    def predict_one(self, raw_features: dict) -> float:
        """
        On-demand inference for a single set of raw (unscaled) indicator values.

        Parameters
        ----------
        raw_features : dict
            Lowercase API keys (iaa, ieg, ips, ida, ipv, ian, inde,
            defasagem, fase_num, gender, age, mat, por, tenure,
            n_av, missing_grades). Missing keys default to 0.0.

        Returns
        -------
        float — risk score in [0, 1].
        """
        if not self._model_repo.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        scaler = self._data_repo.load_scaler()

        # Build feature vector in the exact ETL column order
        row = [float(raw_features.get(api_key, 0.0)) for api_key, _ in _API_TO_FEATURE]
        X_raw = np.array([row], dtype="float32")  # shape (1, 16)
        X_scaled = scaler.transform(X_raw).astype("float32")

        scores = self._model_repo.predict(X_scaled)
        return float(scores[0])
