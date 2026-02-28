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
        Load model + data, run inference, return StudentRecord list.
        Raises on any loading failure (caller decides on retry / 503).
        """
        self._model_repo.load()

        metadata: list[dict[str, Any]] = self._data_repo.load_metadata()
        X: np.ndarray = self._data_repo.load_features()

        log.info("Running batch inference over %d students…", len(X))
        scores: np.ndarray = self._model_repo.predict(X)
        log.info("Inference complete ✓")

        records: list[StudentRecord] = []
        for i, (meta, score) in enumerate(zip(metadata, scores, strict=False)):
            indicators = Indicators(
                iaa=meta.get("iaa"),
                ieg=meta.get("ieg"),
                ips=meta.get("ips"),
                ida=meta.get("ida"),
                ipv=meta.get("ipv"),
                ipp=meta.get("ipp"),
                inde=meta.get("inde"),
                defasagem=meta.get("defasagem"),
                fase_num=meta.get("fase_num"),
            )
            record = StudentRecord.build(
                student_id=i,
                ra=meta.get("ra", f"RA-{i}"),
                display_name=meta.get("display_name", f"Aluno-{i}"),
                phase=meta.get("phase", "N/A"),
                phase_num=meta.get("fase_num", 0),
                class_group=meta.get("class_group", "N/A"),
                year=meta.get("year", 2024),
                risk_score=float(score),
                indicators=indicators,
            )
            records.append(record)

        return records
