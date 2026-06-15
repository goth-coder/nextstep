"""
Precomputed dashboard-scores artifact (``processed/scored_students.json`` on GCS).

Separates *batch scoring* (offline / one cold start) from *serving*. Once the
artifact exists, the API populates its cache from a single JSON blob instead of
loading the model and re-running inference on every cold start — which is the
dominant cost when ``min-instances=0``.

The artifact is regenerated when:
  • it is missing (first cold start after deploy — self-healing), or
  • it is deleted on model promotion (see the Train workflow).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from app.domain.student import StudentRecord

log = logging.getLogger(__name__)

ARTIFACT_BLOB = "processed/scored_students.json"
ARTIFACT_VERSION = 1
GCS_BUCKET = os.getenv("GCS_BUCKET", "")


def _bucket():
    from google.cloud import storage  # noqa: PLC0415

    return storage.Client().bucket(GCS_BUCKET)


def load() -> list[StudentRecord] | None:
    """Return scored records from GCS, or None if absent/stale/unreadable."""
    if not GCS_BUCKET:
        return None
    try:
        blob = _bucket().blob(ARTIFACT_BLOB)
        if not blob.exists():
            log.info("No scores artifact at gs://%s/%s — inference will run", GCS_BUCKET, ARTIFACT_BLOB)
            return None
        payload = json.loads(blob.download_as_bytes())
        if payload.get("version") != ARTIFACT_VERSION:
            log.warning("Scores artifact version=%s != %s — ignoring", payload.get("version"), ARTIFACT_VERSION)
            return None
        records = [StudentRecord.from_dict(d) for d in payload.get("students", [])]
        if not records:
            return None
        log.info(
            "Loaded %d scored students from artifact (generated_at=%s, model=%s)",
            len(records), payload.get("generated_at"), payload.get("model_name"),
        )
        return records
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read scores artifact (%s) — falling back to inference", exc)
        return None


def save(records: list[StudentRecord], model_name: str | None = None) -> bool:
    """Serialize scored records to GCS. Non-fatal: returns False on any failure."""
    if not GCS_BUCKET:
        log.info("GCS_BUCKET not set — skipping scores-artifact persist")
        return False
    if not records:
        return False
    try:
        payload = {
            "version": ARTIFACT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name or os.getenv("MLFLOW_MODEL_NAME", "nextstep-lstm"),
            "count": len(records),
            "students": [r.to_dict() for r in records],
        }
        blob = _bucket().blob(ARTIFACT_BLOB)
        blob.upload_from_string(json.dumps(payload), content_type="application/json")
        log.info("Persisted %d scored students to gs://%s/%s", len(records), GCS_BUCKET, ARTIFACT_BLOB)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to persist scores artifact (%s) — serving still works", exc)
        return False
