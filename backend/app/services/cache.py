"""
StudentCacheService — in-memory store of StudentRecord after batch inference.

Single Responsibility: manage cache lifecycle (load, query).
Replaces the old prediction_cache.py module-level globals with a proper class.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

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
        self._students_ready: bool = False  # Phase 1: metadata loaded
        self._ready: bool = False           # Phase 2: model scores loaded
        self._last_error: str | None = None
        self._last_attempt_at: str | None = None
        self._attempt_count: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_students(self) -> None:
        """Phase 1: load student metadata only (no model required)."""
        self._attempt_count += 1
        self._last_attempt_at = datetime.now(timezone.utc).isoformat()
        records = self._prediction_service.load_students_only()
        self._cache = {r.student_id: r for r in records}
        self._students_ready = True
        self._last_error = None
        log.info("Phase-1 cache ready ✓  %d students (no scores)", len(self._cache))

    def load_model_scores(self) -> None:
        """Phase 2: run model inference and update risk scores in cache."""
        self._last_attempt_at = datetime.now(timezone.utc).isoformat()
        records = self._prediction_service.run_batch_inference()
        self._cache = {r.student_id: r for r in records}
        self._ready = True
        log.info("Phase-2 cache ready ✓  %d students (with scores)", len(self._cache))

    def load(self) -> None:
        """Legacy: run full batch inference (called during retry loop)."""
        self.load_model_scores()

    def load_with_retry(self) -> None:
        """Load cache with bounded retry and exponential backoff."""
        max_attempts = max(1, int(os.getenv("MODEL_LOAD_MAX_ATTEMPTS", "8")))
        base_delay = max(0.0, float(os.getenv("MODEL_LOAD_RETRY_BASE_SECONDS", "2")))
        max_delay = max(base_delay, float(os.getenv("MODEL_LOAD_RETRY_MAX_SECONDS", "30")))

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                self.load()
                return
            except Exception as exc:  # noqa: BLE001
                self._ready = False
                self._last_error = str(exc)
                last_exc = exc
                log.error(
                    "Cache/model load failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    exc,
                    exc_info=True,
                )

                if attempt >= max_attempts:
                    break

                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                log.warning("Retrying cache/model load in %.1fs", delay)
                time.sleep(delay)

        raise RuntimeError(f"Model/cache load failed after {max_attempts} attempts") from last_exc

    # ── Queries ───────────────────────────────────────────────────────────────

    def has_students(self) -> bool:
        """True after phase-1 (metadata loaded, students visible)."""
        return self._students_ready

    def is_ready(self) -> bool:
        """True after phase-2 (model loaded, risk scores populated)."""
        return self._ready

    def get_all(self) -> list[StudentRecord]:
        return list(self._cache.values())

    def get_by_id(self, student_id: int) -> StudentRecord | None:
        return self._cache.get(int(student_id))

    def count(self) -> int:
        return len(self._cache)

    def attempts(self) -> int:
        return self._attempt_count

    def last_error(self) -> str | None:
        return self._last_error

    def last_attempt_at(self) -> str | None:
        return self._last_attempt_at

    def attempts(self) -> int:
        return self._attempt_count

    def predict_one(self, raw_features: dict) -> float:
        """On-demand inference for arbitrary indicator values (bypasses cache)."""
        if not self._ready:
            raise RuntimeError("Model not loaded yet.")
        return self._prediction_service.predict_one(raw_features)
