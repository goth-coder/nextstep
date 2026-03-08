"""
Evaluator — threshold selection and metric computation.

SRP: accepts a trained model + split tensors, returns EvalResult.
No training, no MLflow, no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    threshold: float
    val_f1_internal: float  # F1 used to select threshold (on val set)
    val_auc: float  # ROC-AUC on test set
    val_f1: float  # F1 on test set at selected (val-derived) threshold
    test_f1_oracle: float  # best achievable F1 on test set (oracle threshold) — quality gate metric
    train_loss: float  # final epoch mean loss from TrainingLoop


class PlattCalibrator:
    """
    Platt scaling: fit a 1-feature logistic regression on val logits to remap
    model outputs to calibrated probabilities matching the true class prior.

    Why this is needed
    ------------------
    WeightedRandomSampler trains the LSTM on artificially balanced 50/50 batches.
    The model sigmoid is therefore calibrated for P(y=1) = 0.50, but the real
    prior is ~17 %.  Raw probabilities compress near 0: positive cases score
    0.02-0.05, and the PR-curve F1 is maximised at a threshold like 0.0188.
    A threshold that microscopic is extremely sensitive to the year-to-year
    distribution shift between the val set (22→23) and the test set (23→24).

    After calibration: threshold lives near the true prior (~0.15-0.20), so
    ±0.05 variation from temporal shift causes far smaller F1 degradation.
    """

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

        # C=1.0 (mild L2) prevents overfitting when val set has only ~20 positives
        self._lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        self._is_fitted = False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit on (logits, true labels). Returns self for chaining."""
        self._lr.fit(logits.reshape(-1, 1), y)
        self._is_fitted = True
        a = float(self._lr.coef_[0, 0])
        b = float(self._lr.intercept_[0])
        log.info("PlattCalibrator fitted — A=%.4f  B=%.4f", a, b)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated P(y=1) for each logit."""
        if not self._is_fitted:
            raise RuntimeError("PlattCalibrator.fit() must be called first.")
        return self._lr.predict_proba(logits.reshape(-1, 1))[:, 1]


class Evaluator:
    """
    Selects the optimal decision threshold via PR-curve F1-maximisation
    on the validation set, then computes final metrics on the test set.

    With Platt calibration (recommended use):
        calibrator = evaluator.fit_calibrator(model, X_val, y_val)
        threshold, f1 = evaluator.find_threshold(model, X_val, y_val, calibrator)
        result = evaluator.evaluate(model, X_test, y_test, threshold, f1, loss, calibrator)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def find_threshold(
        self,
        model: torch.nn.Module,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        calibrator: Optional[PlattCalibrator] = None,
    ) -> tuple[float, float]:
        """
        Returns (threshold, val_f1_internal) — the threshold that maximises
        F1 on the validation split.

        Pass a fitted PlattCalibrator to operate on calibrated probabilities
        (recommended — see class docstring).
        """
        probs = self._get_probs(model, X_val, calibrator)
        precision, recall, thresholds = precision_recall_curve(y_val.numpy().astype(int), probs)
        # np.where evaluates BOTH branches before selecting, so the division
        # still runs on zero-denominator elements — suppress the resulting
        # RuntimeWarning explicitly.
        with np.errstate(invalid="ignore"):
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
        calibrator: Optional[PlattCalibrator] = None,
    ) -> EvalResult:
        """Compute AUC + F1 on the test set using the val-derived threshold.

        Also computes `test_f1_oracle` — the best F1 achievable on the test set
        with its own optimal threshold.  This is used for the quality gate: it
        measures whether the model *can* discriminate well, independent of how
        well the val-derived threshold transfers across temporal splits.

        Pass the same PlattCalibrator used in find_threshold so both threshold
        selection and final evaluation operate on the same probability space.
        """
        probs = self._get_probs(model, X_test, calibrator)
        labels = y_test.numpy().astype(int)

        auc = float(roc_auc_score(labels, probs))

        # Deployed F1: val-derived threshold applied to test set
        preds = (probs >= threshold).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))

        # Oracle F1: best possible F1 on test set (calibrated), for quality gate
        _, oracle_f1 = self.find_threshold(model, X_test, y_test, calibrator)

        log.info(
            "Test AUC=%.4f  F1(deployed)=%.4f  F1(oracle)=%.4f  threshold=%.4f",
            auc,
            f1,
            oracle_f1,
            threshold,
        )
        if oracle_f1 > 0 and f1 < oracle_f1 * 0.75:
            log.warning(
                "Threshold transfer degradation detected: deployed F1=%.4f vs oracle F1=%.4f — "
                "consider recalibrating threshold on recent data.",
                f1,
                oracle_f1,
            )
        return EvalResult(
            threshold=threshold,
            val_f1_internal=val_f1_internal,
            val_auc=auc,
            val_f1=f1,
            test_f1_oracle=oracle_f1,
            train_loss=train_loss,
        )

    def fit_calibrator(
        self,
        model: torch.nn.Module,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> PlattCalibrator:
        """
        Fit a PlattCalibrator on the validation-set logits.

        Call AFTER TrainingLoop.fit(), BEFORE find_threshold():
            calibrator = evaluator.fit_calibrator(model, X_val, y_val)
            threshold, f1 = evaluator.find_threshold(model, X_val, y_val, calibrator)
        """
        logits = self._get_logits(model, X_val)
        calibrator = PlattCalibrator()
        calibrator.fit(logits, y_val.numpy().astype(int))
        return calibrator

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_logits(model: torch.nn.Module, X: torch.Tensor) -> np.ndarray:
        """Return raw model logits as a flat numpy array (no sigmoid)."""
        model.eval()
        with torch.no_grad():
            return model(X).squeeze(-1).numpy()

    def _get_probs(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        calibrator: Optional[PlattCalibrator] = None,
    ) -> np.ndarray:
        """Return probabilities: calibrated if calibrator provided, else raw sigmoid."""
        logits = self._get_logits(model, X)
        if calibrator is not None:
            return calibrator.predict_proba(logits)
        # Fallback: numerically stable sigmoid without torch dependency
        return 1.0 / (1.0 + np.exp(-logits))
