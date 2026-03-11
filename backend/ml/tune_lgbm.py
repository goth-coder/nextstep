"""
tune_lgbm.py — HPO for LightGBM using Optuna + MLflow nested runs.

Usage:
    python ml/tune_lgbm.py --trials 30
    python ml/tune_lgbm.py --trials 10 --no-train

Steps:
    1. Load processed arrays from data/processed/
    2. Open parent MLflow run in experiment "nextstep-lgbm-hpo"
    3. Run N Optuna trials (each = child MLflow run, calibrated val F1 objective)
    4. Save best_params_lgbm.json to data/processed/
    5. Unless --no-train: retrain with best params and register @staging + @prod
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from train_lgbm import (  # noqa: E402
    DEFAULT_PARAMS,
    MLFLOW_URI,
    PROCESSED_DIR,
    _log_run,
    find_threshold,
    fit_calibrator,
    load_arrays,
)

# ── Configuration ─────────────────────────────────────────────────────────────
HPO_EXPERIMENT = os.getenv("MLFLOW_HPO_EXPERIMENT", "nextstep-lgbm-hpo")
BEST_PARAMS_PATH = PROCESSED_DIR / "best_params_lgbm.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("tune_lgbm")


# ── Optuna objective ──────────────────────────────────────────────────────────

class _Objective:
    """Closure that captures train/val data and parent run for Optuna."""

    def __init__(
        self,
        X_tr, y_tr, X_val, y_val,
        parent_run_id: str,
        n_trials: int,
    ) -> None:
        self._X_tr = X_tr
        self._y_tr = y_tr
        self._X_val = X_val
        self._y_val = y_val
        self._parent_run_id = parent_run_id
        self._n_trials = n_trials
        self._best_f1 = 0.0

    def __call__(self, trial: optuna.Trial) -> float:
        import lightgbm as lgb

        # n_estimators is NOT tuned: early stopping controls convergence,
        # which decouples it from learning_rate and avoids undertrained models.
        # lr lower-bound raised to 1e-2 to avoid ultra-slow runs that need
        # thousands of trees to converge (search space was 1e-3 before).
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "n_estimators": 4000,  # ceiling; early stopping decides the actual count
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

        log.info(
            "[Trial %d/%d] leaves=%d  depth=%d  lr=%.4f  spw=%.2f",
            trial.number + 1, self._n_trials,
            params["num_leaves"], params["max_depth"], params["learning_rate"],
            params["scale_pos_weight"],
        )

        model = lgb.LGBMClassifier(**params)
        model.fit(
            self._X_tr, self._y_tr.astype(int),
            eval_set=[(self._X_val, self._y_val.astype(int))],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        actual_trees = model.best_iteration_ or model.n_estimators_
        log.info("[Trial %d/%d] early-stopped at %d trees", trial.number + 1, self._n_trials, actual_trees)

        calibrator = fit_calibrator(model, self._X_val, self._y_val)
        threshold, val_f1 = find_threshold(model, self._X_val, self._y_val, calibrator)

        is_best = val_f1 > self._best_f1
        if is_best:
            self._best_f1 = val_f1
        log.info(
            "[Trial %d/%d] val_f1=%.4f  threshold=%.4f%s",
            trial.number + 1, self._n_trials, val_f1, threshold,
            "  ← NEW BEST" if is_best else "",
        )

        # Log trial as child MLflow run; replace ceiling with actual trees used
        logged_params = {**params, "n_estimators": model.best_iteration_ or model.n_estimators_}
        from training.evaluator import EvalResult
        trial_result = EvalResult(
            threshold=threshold,
            val_f1_internal=val_f1,
            val_auc=0.0,  # not computed per-trial to save time
            val_f1=0.0,
            test_f1_oracle=0.0,
            train_loss=0.0,
        )
        _log_run(
            params=logged_params,
            result=trial_result,
            model=model,
            calibrator=calibrator,
            parent_run_id=self._parent_run_id,
            run_name=f"lgbm-trial-{trial.number}",
            experiment_name=HPO_EXPERIMENT,
        )

        log.info("Trial %d — val_f1=%.4f  threshold=%.4f", trial.number, val_f1, threshold)
        return val_f1

def main() -> None:
    parser = argparse.ArgumentParser(description="HPO for LightGBM risk model")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--no-train", action="store_true", help="Skip final retraining after HPO")
    args = parser.parse_args()

    X_tr, y_tr, X_val, y_val, X_test, y_test = load_arrays()
    log.info(
        "Loaded arrays — input_size=%d  train=%d  val=%d  test=%d",
        X_tr.shape[1], len(X_tr), len(X_val), len(X_test),
    )

    # ── HPO phase ─────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(HPO_EXPERIMENT)

    with mlflow.start_run(run_name=f"lgbm-hpo-{args.trials}-trials") as parent_run:
        parent_run_id = parent_run.info.run_id
        log.info("Started HPO parent run: %s", parent_run_id)
        log.info("━" * 60)
        log.info("Running %d Optuna trials (TPE sampler, seed=42)", args.trials)
        log.info("Objective: maximize calibrated val F1  |  val set: %d samples", len(X_val))
        log.info("━" * 60)

        objective = _Objective(X_tr, y_tr, X_val, y_val, parent_run_id, args.trials)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    log.info("━" * 60)
    log.info(
        "HPO complete — best trial #%d  val_f1=%.4f",
        study.best_trial.number + 1, study.best_value,
    )
    log.info("Best hyperparameters:")
    for k, v in study.best_params.items():
        log.info("  %-22s = %s", k, v)
    log.info("━" * 60)

    # n_estimators is not part of study.best_params (uses early stopping);
    # set ceiling high so final train_lgbm also relies on early stopping.
    best_params = {**DEFAULT_PARAMS, **study.best_params, "n_estimators": 2000}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
    log.info("Best params saved to %s", BEST_PARAMS_PATH)

    # ── Final retrain with best params ────────────────────────────────────────
    if args.no_train:
        log.info("--no-train set — skipping final training run")
        return

    log.info("━" * 60)
    log.info("Retraining final model with best HPO params ...")
    log.info("━" * 60)
    from train_lgbm import train_lgbm  # noqa: E402
    train_lgbm(params=best_params)


if __name__ == "__main__":
    main()
