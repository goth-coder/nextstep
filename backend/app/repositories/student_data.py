"""
Student data repository — reads ETL outputs from disk.

Implements StudentDataRepository port.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = _BACKEND_DIR / "data" / "processed"


class DiskStudentDataRepository:
    """Reads students_meta.pkl and X_inference.npy written by data_loader.py."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        self._dir = processed_dir

    # ── StudentDataRepository protocol ────────────────────────────────────────

    def load_metadata(self) -> list[dict[str, Any]]:
        path = self._dir / "students_meta.pkl"
        if not path.exists():
            raise FileNotFoundError(f"students_meta.pkl not found at {path}. Run 'python ml/data_loader.py' first.")
        with open(path, "rb") as f:
            meta: list[dict] = pickle.load(f)
        log.info("Loaded metadata for %d students", len(meta))
        return meta

    def load_features(self) -> np.ndarray:
        """Load the inference feature matrix (2024 students, already scaled)."""
        path = self._dir / "X_inference.npy"
        if not path.exists():
            raise FileNotFoundError(f"X_inference.npy not found at {path}. Run 'python ml/data_loader.py' first.")
        X = np.load(path)
        log.info("Loaded inference features: shape=%s", X.shape)
        return X

    def load_scaler(self):
        """Load the RobustScaler fitted during ETL (used for on-demand /predict)."""
        path = self._dir / "scaler.pkl"
        if not path.exists():
            raise FileNotFoundError(f"scaler.pkl not found at {path}. Run 'python ml/data_loader.py' first.")
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        log.info("Loaded scaler: %s", type(scaler).__name__)
        return scaler
