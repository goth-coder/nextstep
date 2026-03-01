"""
Evaluator — threshold selection and metric computation.

SRP: accepts a trained model + split tensors, returns EvalResult.
No training, no MLflow, no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    threshold: float
    val_f1_internal: float  # F1 used to select threshold (on val set)
    val_auc: float  # ROC-AUC on test set
    val_f1: float  # F1 on test set at selected threshold
    train_loss: float  # final epoch mean loss from TrainingLoop


class Evaluator:
    """
    Selects the optimal decision threshold via PR-curve F1-maximisation
    on the validation set, then computes final metrics on the test set.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def find_threshold(
        self,
        model: torch.nn.Module,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Returns (threshold, val_f1_internal) — the threshold that maximises
        F1 on the validation split.
        """
        probs = self._predict_proba(model, X_val)
        precision, recall, thresholds = precision_recall_curve(y_val.numpy().astype(int), probs)
        # f1 shape matches thresholds; precision/recall have one extra element
        f1s = np.where(
            (precision[:-1] + recall[:-1]) > 0,
            2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
            0.0,
        )
        best_idx = int(np.argmax(f1s))
        threshold = float(thresholds[best_idx])
        val_f1 = float(f1s[best_idx])
        log.info("Best val threshold=%.4f  val_f1=%.4f", threshold, val_f1)
        return threshold, val_f1

    def evaluate(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        threshold: float,
        val_f1_internal: float,
        train_loss: float,
    ) -> EvalResult:
        """Compute AUC + F1 on the test set using the val-derived threshold."""
        probs = self._predict_proba(model, X_test)
        labels = y_test.numpy().astype(int)

        auc = float(roc_auc_score(labels, probs))
        preds = (probs >= threshold).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))

        log.info("Test AUC=%.4f  F1=%.4f  threshold=%.4f", auc, f1, threshold)
        return EvalResult(
            threshold=threshold,
            val_f1_internal=val_f1_internal,
            val_auc=auc,
            val_f1=f1,
            train_loss=train_loss,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _predict_proba(model: torch.nn.Module, X: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(X).squeeze(-1)
            return torch.sigmoid(logits).numpy()
