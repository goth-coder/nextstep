"""
StudentCacheService — in-memory store of StudentRecord after batch inference.

Single Responsibility: manage cache lifecycle (load, query).
Replaces the old prediction_cache.py module-level globals with a proper class.
"""

from __future__ import annotations

import logging

from app.domain.student import StudentRecord
from app.services.prediction import PredictionService

log = logging.getLogger(__name__)


class StudentCacheService:
    """
    Holds all StudentRecords in memory after startup inference.

    Designed to be instantiated once (Flask app context) and injected
    into routes via the module-level shim in prediction_cache.py.
    """

    def __init__(self, prediction_service: PredictionService) -> None:
        self._prediction_service = prediction_service
        self._cache: dict[int, StudentRecord] = {}
        self._ready: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Run batch inference and populate cache. Called once at app startup."""
        records = self._prediction_service.run_batch_inference()
        self._cache = {r.student_id: r for r in records}
        self._ready = True
        log.info("StudentCacheService ready ✓  %d students loaded", len(self._cache))

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def get_all(self) -> list[StudentRecord]:
        return list(self._cache.values())

    def get_by_id(self, student_id: int) -> StudentRecord | None:
        return self._cache.get(int(student_id))

    def count(self) -> int:
        return len(self._cache)

    def predict_one(self, raw_features: dict) -> float:
        """On-demand inference for arbitrary indicator values (bypasses cache)."""
        if not self._ready:
            raise RuntimeError("Model not loaded yet.")
        return self._prediction_service.predict_one(raw_features)
