"""
Student data repository — reads ETL outputs from GCS (primary) or local disk (dev).

Implements StudentDataRepository port.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = _BACKEND_DIR / "data" / "processed"
GCS_BUCKET = os.getenv("GCS_BUCKET", "")


def _gcs_download_bytes(blob_name: str) -> bytes:
    """Download a blob from GCS_BUCKET and return its raw bytes."""
    from google.cloud import storage  # noqa: PLC0415

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    log.info("Downloaded gs://%s/%s (%d bytes)", GCS_BUCKET, blob_name, len(data))
    return data


class GCSStudentDataRepository:
    """Reads students_meta.pkl and X_inference.npy from GCS (primary store)."""

    def load_metadata(self) -> list[dict[str, Any]]:
        raw = _gcs_download_bytes("processed/students_meta.pkl")
        meta: list[dict] = pickle.loads(raw)  # noqa: S301
        log.info("Loaded metadata for %d students from GCS", len(meta))
        return meta

    def load_features(self) -> np.ndarray:
        raw = _gcs_download_bytes("processed/X_inference.npy")
        X = np.load(io.BytesIO(raw))
        log.info("Loaded inference features from GCS: shape=%s", X.shape)
        return X

    def load_scaler(self):
        raw = _gcs_download_bytes("processed/scaler.pkl")
        scaler = pickle.loads(raw)  # noqa: S301
        log.info("Loaded scaler from GCS: %s", type(scaler).__name__)
        return scaler

    def load_test_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        X_raw = _gcs_download_bytes("processed/X_test.npy")
        y_raw = _gcs_download_bytes("processed/y_test.npy")
        return np.load(io.BytesIO(X_raw)), np.load(io.BytesIO(y_raw))


class DiskStudentDataRepository:
    """Reads ETL outputs from local disk (dev / CI fallback)."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        self._dir = processed_dir

    def _require(self, name: str) -> Path:
        p = self._dir / name
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p}. Run 'python ml/data_loader.py' first.")
        return p

    def load_metadata(self) -> list[dict[str, Any]]:
        with open(self._require("students_meta.pkl"), "rb") as f:
            meta: list[dict] = pickle.load(f)  # noqa: S301
        log.info("Loaded metadata for %d students", len(meta))
        return meta

    def load_features(self) -> np.ndarray:
        X = np.load(self._require("X_inference.npy"))
        log.info("Loaded inference features: shape=%s", X.shape)
        return X

    def load_scaler(self):
        with open(self._require("scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)  # noqa: S301
        log.info("Loaded scaler: %s", type(scaler).__name__)
        return scaler

    def load_test_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.load(self._require("X_test.npy")), np.load(self._require("y_test.npy"))


def make_student_data_repository():
    """Factory: return GCS repo if bucket is configured, else local disk repo."""
    if GCS_BUCKET:
        log.info("Using GCSStudentDataRepository (bucket=%s)", GCS_BUCKET)
        return GCSStudentDataRepository()
    log.warning("GCS_BUCKET not set — using DiskStudentDataRepository (local dev)")
    return DiskStudentDataRepository()
