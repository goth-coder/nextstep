#!/bin/bash
set -e

DB_PATH="/tmp/mlflow.db"
ARTIFACT_DIR="/tmp/mlruns"
GCS="python3 /sync.py cp"

echo "MLflow starting — bucket: '${MLFLOW_BUCKET}'"

if [[ -n "${MLFLOW_BUCKET}" ]]; then
  GCS_DB="gs://${MLFLOW_BUCKET}/mlflow/mlflow.db"

  # Restore existing DB from GCS if available
  if $GCS "$GCS_DB" "$DB_PATH" 2>/dev/null; then
    echo "✅ Restored MLflow DB from GCS"
  else
    echo "No existing DB in GCS — starting fresh"
  fi

  # Background sync: save DB to GCS every SYNC_INTERVAL seconds
  _sync_loop() {
    while true; do
      sleep "${SYNC_INTERVAL}"
      if $GCS "$DB_PATH" "$GCS_DB" 2>/dev/null; then
        echo "[sync] MLflow DB backed up to GCS"
      else
        echo "[sync] WARNING: GCS backup failed"
      fi
    done
  }
  _sync_loop &

  exec mlflow server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --backend-store-uri "sqlite:////${DB_PATH}" \
    --serve-artifacts \
    --artifacts-destination "gs://${MLFLOW_BUCKET}/mlflow/artifacts" \
    --default-artifact-root "mlflow-artifacts:/"
else
  echo "⚠️  MLFLOW_BUCKET not set — running in local-only mode (no GCS persistence)"
  mkdir -p "$ARTIFACT_DIR"
  exec mlflow server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --backend-store-uri "sqlite:////${DB_PATH}" \
    --serve-artifacts \
    --artifacts-destination "${ARTIFACT_DIR}" \
    --default-artifact-root "mlflow-artifacts:/"
fi
