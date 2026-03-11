"""
train_lgbm.py — LightGBM training entrypoint.

Usage:
    python ml/train_lgbm.py
    python ml/train_lgbm.py --config data/processed/best_params_lgbm.json

Steps:
    1. Load processed arrays from data/processed/
    2. Load config (JSON or defaults)
    3. Fit LGBMClassifier
    4. Fit PlattCalibrator  (log-odds of LGBM probs → logistic regression)
    5. find_threshold()     on val set (calibrated probs)
    6. evaluate()           on test set
    7. Quality gate
    8. log to MLflow as nextstep-lgbm + promote @staging / @prod
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

# Silence: "Found torch version ... contains a local version label"
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # backend/
sys.path.insert(0, str(ROOT / "ml"))

from training.evaluator import EvalResult, PlattCalibrator  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "nextstep-lgbm")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "nextstep-lgbm")

MIN_AUC = float(os.getenv("QUALITY_MIN_AUC", "0.70"))
MIN_F1 = float(os.getenv("QUALITY_MIN_F1", "0.48"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("train_lgbm")


# ── Default hyperparameters ───────────────────────────────────────────────────

DEFAULT_PARAMS: dict = {
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 4000,  # ceiling; early stopping controls the actual count
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 5.0,   # ~ neg/pos ratio baseline
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_arrays() -> tuple[np.ndarray, ...]:
    """Load processed .npy files and return train/val/test splits (numpy)."""

    def _npy(name: str) -> np.ndarray:
        p = PROCESSED_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p} — run data_loader.py first")
        return np.load(p)

    X_train_np = _npy("X_train.npy")
    y_train_np = _npy("y_train.npy")
    X_test_np = _npy("X_test.npy")
    y_test_np = _npy("y_test.npy")

    # Temporal val split: last 20% of train (same as LSTM entrypoint)
    val_size = max(1, int(0.20 * len(X_train_np)))
    tr_size = len(X_train_np) - val_size

    X_tr, X_val = X_train_np[:tr_size], X_train_np[tr_size:]
    y_tr, y_val = y_train_np[:tr_size], y_train_np[tr_size:]

    log.info(
        "Splits — train=%d (pos=%.1f%%)  val=%d (pos=%.1f%%)  test=%d (pos=%.1f%%)",
        tr_size, float(y_tr.mean()) * 100,
        val_size, float(y_val.mean()) * 100,
        len(y_test_np), float(y_test_np.mean()) * 100,
    )
    return X_tr, y_tr, X_val, y_val, X_test_np, y_test_np


# ── Calibration helpers ───────────────────────────────────────────────────────

def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Convert LGBM predict_proba output to log-odds for PlattCalibrator."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))


def fit_calibrator(model: lgb.LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray) -> PlattCalibrator:
    probs = model.predict_proba(X_val)[:, 1]
    logits = _probs_to_logits(probs)
    cal = PlattCalibrator()
    cal.fit(logits, y_val.astype(int))
    return cal


def _get_calibrated_probs(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    calibrator: PlattCalibrator | None,
) -> np.ndarray:
    probs = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        logits = _probs_to_logits(probs)
        return calibrator.predict_proba(logits)
    return probs


# ── Threshold selection + evaluation ─────────────────────────────────────────

def find_threshold(
    model: lgb.LGBMClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    calibrator: PlattCalibrator | None = None,
) -> tuple[float, float]:
    """PR-curve F1 maximisation on the validation set. Returns (threshold, val_f1)."""
    probs = _get_calibrated_probs(model, X_val, calibrator)
    precision, recall, thresholds = precision_recall_curve(y_val.astype(int), probs)
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
    model: lgb.LGBMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    val_f1_internal: float,
    calibrator: PlattCalibrator | None = None,
) -> EvalResult:
    probs = _get_calibrated_probs(model, X_test, calibrator)
    labels = y_test.astype(int)

    auc = float(roc_auc_score(labels, probs))
    preds = (probs >= threshold).astype(int)
    f1 = float(f1_score(labels, preds, zero_division=0))

    # Oracle F1 on test set (best achievable) — quality gate metric
    _, oracle_f1 = find_threshold(model, X_test, y_test, calibrator)

    log.info(
        "Test AUC=%.4f  F1(deployed)=%.4f  F1(oracle)=%.4f  threshold=%.4f",
        auc, f1, oracle_f1, threshold,
    )
    return EvalResult(
        threshold=threshold,
        val_f1_internal=val_f1_internal,
        val_auc=auc,
        val_f1=f1,
        test_f1_oracle=oracle_f1,
        train_loss=0.0,  # LGBM does not report BCE loss per epoch
    )


# ── MLflow logging ────────────────────────────────────────────────────────────

def _log_run(
    params: dict,
    result: EvalResult,
    model: lgb.LGBMClassifier,
    calibrator: PlattCalibrator | None,
    parent_run_id: str | None = None,
    run_name: str = "lgbm-train",
    input_size: int | None = None,
    experiment_name: str | None = None,
) -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name or EXPERIMENT_NAME)

    kwargs: dict = {"run_name": run_name, "nested": parent_run_id is not None}
    if parent_run_id:
        kwargs["tags"] = {"mlflow.parentRunId": parent_run_id}

    with mlflow.start_run(**kwargs) as run:
        run_id = run.info.run_id
        mlflow.set_tag("model_type", "lgbm")

        mlflow.log_params({str(k): str(v) for k, v in params.items()})
        if input_size is not None:
            mlflow.log_param("input_size", str(input_size))

        metrics: dict[str, float] = {
            "val_f1_internal": result.val_f1_internal,
            "threshold": result.threshold,
        }
        if result.val_auc > 0.0:
            metrics["val_auc"] = result.val_auc
        if result.val_f1 > 0.0:
            metrics["val_f1"] = result.val_f1
        if result.test_f1_oracle > 0.0:
            metrics["test_f1_oracle"] = result.test_f1_oracle
        mlflow.log_metrics(metrics)

        # Log scaler artifact if it exists
        if SCALER_PATH.exists():
            mlflow.log_artifact(str(SCALER_PATH), artifact_path="scaler")

        # Log LGBM model
        mlflow.lightgbm.log_model(model, artifact_path="model")

        # Log calibrator
        if calibrator is not None:
            with tempfile.TemporaryDirectory() as tmp:
                cal_path = Path(tmp) / "calibrator.pkl"
                with open(cal_path, "wb") as _f:
                    pickle.dump(calibrator, _f)
                mlflow.log_artifact(str(cal_path), artifact_path="calibrator")
                log.info("Calibrator saved to MLflow run %s", run_id)

    log.info("MLflow run logged: %s", run_id)
    return run_id


def _promote(run_id: str, aliases: list[str]) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    # Register the model (creates if not exists)
    uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(uri, MODEL_NAME)
    version = mv.version
    log.info("Registered model '%s' version %s", MODEL_NAME, version)
    for alias in aliases:
        client.set_registered_model_alias(MODEL_NAME, alias, version)
        log.info("Alias @%s → version %s", alias, version)


# ── Public entry-point (used by tune_lgbm.py) ────────────────────────────────

def train_lgbm(
    params: dict | None = None,
    parent_run_id: str | None = None,
) -> EvalResult:
    """Train, calibrate, evaluate and register one LGBM model. Returns EvalResult."""
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_arrays()

    # Compute class imbalance from actual training data unless caller overrides
    neg = float((y_tr == 0).sum())
    pos = float((y_tr == 1).sum())
    dynamic_spw = neg / pos if pos > 0 else 5.0
    defaults_with_spw = {**DEFAULT_PARAMS, "scale_pos_weight": dynamic_spw}
    p = {**defaults_with_spw, **(params or {})}

    log.info(
        "Class balance — neg=%d  pos=%d  scale_pos_weight=%.2f",
        int(neg), int(pos), p["scale_pos_weight"],
    )
    log.info("Training LGBMClassifier — params=%s", p)
    model = lgb.LGBMClassifier(**p)
    model.fit(
        X_tr, y_tr.astype(int),
        eval_set=[(X_val, y_val.astype(int))],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    actual_trees = model.best_iteration_ or model.n_estimators_
    log.info("Fit complete — %d trees grown (early stopping)", actual_trees)

    log.info("Fitting Platt calibrator on val set (%d samples) ...", len(X_val))
    calibrator = fit_calibrator(model, X_val, y_val)
    log.info("Searching optimal threshold via PR-curve on val set ...")
    threshold, val_f1_internal = find_threshold(model, X_val, y_val, calibrator)
    log.info("Evaluating on test set (%d samples) ...", len(X_test))
    result = evaluate(model, X_test, y_test, threshold, val_f1_internal, calibrator)

    # Quality gate
    if result.val_auc < MIN_AUC or result.test_f1_oracle < MIN_F1:
        log.error(
            "Quality gate FAILED — AUC=%.4f (min %.2f)  F1_oracle=%.4f (min %.2f)  F1_deployed=%.4f",
            result.val_auc, MIN_AUC, result.test_f1_oracle, MIN_F1, result.val_f1,
        )
        sys.exit(1)

    log.info(
        "Quality gate PASSED — AUC=%.4f  F1_oracle=%.4f  F1_deployed=%.4f  threshold=%.4f",
        result.val_auc, result.test_f1_oracle, result.val_f1, result.threshold,
    )

    log.info("Logging run to MLflow experiment '%s' ...", EXPERIMENT_NAME)
    run_id = _log_run(
        params=p,
        result=result,
        model=model,
        calibrator=calibrator,
        parent_run_id=parent_run_id,
        input_size=X_tr.shape[1],
    )
    _promote(run_id, aliases=["staging", "prod"])
    log.info("━" * 60)
    log.info("Training complete ✓  run_id=%s", run_id)
    log.info("  AUC=%.4f   F1_oracle=%.4f   threshold=%.4f", result.val_auc, result.test_f1_oracle, result.threshold)
    log.info("  Model registered as '%s' @staging @prod", MODEL_NAME)
    log.info("━" * 60)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM risk model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON file with hyperparameters (e.g. best_params_lgbm.json)",
    )
    args = parser.parse_args()

    params: dict | None = None
    if args.config and args.config.exists():
        log.info("Loading params from %s", args.config)
        params = json.loads(args.config.read_text())

    train_lgbm(params=params)
