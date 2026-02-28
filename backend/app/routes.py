"""
Flask Blueprint: REST API routes.

Endpoints:
    GET /health
    GET /api/students
    GET /api/students/<int:student_id>
    GET /api/students/<int:student_id>/advice
    GET /api/model
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import mlflow
from flask import Blueprint, current_app, jsonify

log = logging.getLogger(__name__)
routes_bp = Blueprint("routes", __name__)


@routes_bp.get("/health")
def health():
    cache = current_app.extensions["cache"]
    return jsonify({"status": "ok", "model_loaded": cache.is_ready(), "student_count": cache.count()}), 200


@routes_bp.get("/api/students")
def list_students():
    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        return jsonify({"error": "Model not yet loaded"}), 503
    records = sorted(cache.get_all(), key=lambda r: (-r.risk_score, r.display_name))
    students = [
        {
            "student_id": r.student_id,
            "display_name": r.display_name,
            "phase": r.phase,
            "class_group": r.class_group,
            "risk_score": r.risk_score,
            "risk_tier": r.risk_tier.value,
        }
        for r in records
    ]
    return jsonify({"students": students, "total": len(students)}), 200


@routes_bp.get("/api/students/<int:student_id>")
def get_student(student_id: int):
    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        return jsonify({"error": "Model not yet loaded"}), 503
    record = cache.get_by_id(student_id)
    if record is None:
        return jsonify({"error": "Student not found"}), 404
    ind = record.indicators
    return jsonify(
        {
            "student_id": record.student_id,
            "display_name": record.display_name,
            "phase": record.phase,
            "fase_num": record.phase_num,
            "class_group": record.class_group,
            "risk_score": record.risk_score,
            "risk_tier": record.risk_tier.value,
            "indicators": {
                "iaa": ind.iaa,
                "ieg": ind.ieg,
                "ips": ind.ips,
                "ida": ind.ida,
                "ipv": ind.ipv,
                "ipp": ind.ipp,
                "inde": ind.inde,
                "defasagem": ind.defasagem,
            },
        }
    ), 200


@routes_bp.get("/api/students/<int:student_id>/advice")
def get_advice(student_id: int):
    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        return jsonify({"error": "Model not yet loaded"}), 503
    record = cache.get_by_id(student_id)
    if record is None:
        return jsonify({"error": "Student not found"}), 404
    llm = current_app.extensions["llm"]
    ind = record.indicators
    advice_text, is_fallback = llm.generate_advice(
        display_name=record.display_name,
        indicators={
            "iaa": ind.iaa,
            "ieg": ind.ieg,
            "ips": ind.ips,
            "ida": ind.ida,
            "ipv": ind.ipv,
            "ipp": ind.ipp,
            "inde": ind.inde,
            "defasagem": ind.defasagem,
        },
        risk_score=record.risk_score,
    )
    generated_at = None if is_fallback else datetime.now(timezone.utc).isoformat()
    return jsonify(
        {
            "student_id": record.student_id,
            "advice": advice_text,
            "is_fallback": is_fallback,
            "generated_at": generated_at,
        }
    ), 200


@routes_bp.get("/api/model")
def get_model_info():
    """Return metadata of the currently loaded model from MLflow."""
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Find the @Production version; fall back to highest version number
        model_name = os.getenv("MLFLOW_MODEL_NAME", "nextstep-lstm")
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return jsonify({"error": "No registered model found"}), 404

        # Prefer Production aliases; fall back to latest version
        prod = next((v for v in versions if v.current_stage == "Production"), None)
        target = prod or sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

        run = client.get_run(target.run_id)

        # Parse log-model history for size & flavors
        import json as _json

        history = _json.loads(run.data.tags.get("mlflow.log-model.history", "[]"))
        artifact = history[0] if history else {}
        flavors = artifact.get("flavors", {})
        pytorch_flavor = flavors.get("pytorch", {})

        return jsonify(
            {
                "model_name": model_name,
                "version": target.version,
                "stage": target.current_stage,
                "run_id": target.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "created_at": target.creation_timestamp,
                "trained_at": artifact.get("utc_time_created", ""),
                "source_script": run.data.tags.get("mlflow.source.name", ""),
                "model_size_bytes": artifact.get("model_size_bytes"),
                "pytorch_version": pytorch_flavor.get("pytorch_version", ""),
                "python_version": flavors.get("python_function", {}).get("python_version", ""),
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            }
        ), 200
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to fetch model info: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
