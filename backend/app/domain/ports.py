"""
Port interfaces (Dependency Inversion Principle).

Define contracts that the application layer depends on.
Concrete implementations live in app/repositories/.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelRepository(Protocol):
    """Abstraction over ML model loading and inference."""

    def load(self) -> None:
        """Load/initialise the model. Must be called before predict()."""
        ...

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """
        Run inference.

        Parameters
        ----------
        X : np.ndarray of shape (N, input_size)
            Pre-scaled feature matrix.

        Returns
        -------
        np.ndarray of shape (N,)
            Risk scores in [0, 1].
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """Return True once load() has completed successfully."""
        ...


@runtime_checkable
class StudentDataRepository(Protocol):
    """Abstraction over reading processed student data from disk."""

    def load_metadata(self) -> list[dict[str, Any]]:
        """Return list of student metadata dicts (raw, pre-inference)."""
        ...

    def load_features(self) -> "np.ndarray":
        """Return feature matrix X of shape (N, input_size) for inference."""
        ...
