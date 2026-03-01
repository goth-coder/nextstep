"""
MLflowRegistry — log params/metrics/artifacts, register models, set aliases.

SRP: knows nothing about PyTorch training or data loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from .evaluator import EvalResult
from .trainer import TrainConfig

log = logging.getLogger(__name__)


class MLflowRegistry:
    """
    Thin wrapper around the MLflow Tracking + Registry APIs.

    Usage
    -----
    registry = MLflowRegistry(tracking_uri, experiment_name, model_name)

    # For a single training run:
    run_id = registry.log_run(config, eval_result, model, scaler_path)
    registry.promote(run_id, aliases=["staging", "prod"])

    # For HPO with nested runs:
    parent_run_id = registry.start_parent_run("hpo-sweep")
    ...  # each trial calls registry.log_run(..., parent_run_id=parent_run_id)
    registry.end_parent_run()
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        model_name: str,
    ) -> None:
        self._model_name = model_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._client = MlflowClient(tracking_uri=tracking_uri)

    # ── Parent run (HPO grouping) ─────────────────────────────────────────────

    def start_parent_run(self, run_name: str) -> str:
        """Open a parent MLflow run and return its run_id."""
        self._parent_run = mlflow.start_run(run_name=run_name)
        run_id = self._parent_run.info.run_id
        log.info("Started parent MLflow run %s  id=%s", run_name, run_id)
        return run_id

    def end_parent_run(self) -> None:
        mlflow.end_run()

    # ── Child / standalone run ────────────────────────────────────────────────

    def log_run(
        self,
        config: TrainConfig,
        result: EvalResult,
        model: "torch.nn.Module",
        scaler_path: Optional[Path] = None,
        parent_run_id: Optional[str] = None,
        run_name: str = "train",
        input_size: Optional[int] = None,
    ) -> str:
        """
        Log a single training run to MLflow.

        Returns the run_id of the newly created run.
        The run is ended before returning.
        """
        kwargs: dict = {"run_name": run_name, "nested": parent_run_id is not None}
        if parent_run_id:
            kwargs["tags"] = {"mlflow.parentRunId": parent_run_id}

        with mlflow.start_run(**kwargs) as run:
            run_id = run.info.run_id

            # Params
            mlflow.log_params(config.to_mlflow_params())
            if input_size is not None:
                mlflow.log_param("input_size", str(input_size))

            # Metrics — names kept consistent with frontend expectations
            metrics: dict[str, float] = {
                "train_loss": result.train_loss,
                "val_f1_internal": result.val_f1_internal,  # threshold-selection F1 (val set)
                "threshold": result.threshold,
            }
            # Only log test metrics when they were actually computed (not HPO trials)
            if result.val_auc > 0.0:
                metrics["val_auc"] = result.val_auc
            if result.val_f1 > 0.0:
                metrics["val_f1"] = result.val_f1
            if result.test_f1_oracle > 0.0:
                metrics["test_f1_oracle"] = result.test_f1_oracle
            mlflow.log_metrics(metrics)

            # Scaler artifact (optional)
            if scaler_path and Path(scaler_path).exists():
                mlflow.log_artifact(str(scaler_path), artifact_path="scaler")

            # Model
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=self._model_name,
            )

        log.info("Logged run %s — AUC=%.4f  F1=%.4f", run_id, result.val_auc, result.val_f1)
        return run_id

    # ── Alias promotion ───────────────────────────────────────────────────────

    def promote(
        self,
        run_id: str,
        aliases: Sequence[str] = ("staging", "prod"),
    ) -> None:
        """
        Find the registered model version created by `run_id` and
        assign the given aliases to it.
        """
        versions = self._client.search_model_versions(filter_string=f"run_id='{run_id}' and name='{self._model_name}'")
        if not versions:
            log.warning("No registered version found for run_id=%s", run_id)
            return

        version_number = versions[0].version
        for alias in aliases:
            self._client.set_registered_model_alias(self._model_name, alias, version_number)
            log.info("Set alias '%s' → %s v%s", alias, self._model_name, version_number)

        aliases_str = " + ".join(f"@{a}" for a in aliases)
        log.info(
            "\n"
            "  ✅  Model promoted: %s v%s  [%s]\n"
            "  🚀  Next API restart / deploy will load this version automatically.\n"
            "      (StudentCacheService.load() fetches alias @prod from MLflow at startup)",
            self._model_name,
            version_number,
            aliases_str,
        )
