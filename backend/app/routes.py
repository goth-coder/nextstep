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
            "gender": record.gender,
            "age": record.age,
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


@routes_bp.get("/api/model/drift")
def get_model_drift():
    """Return risk score distribution stats for operational drift monitoring."""
    import math
    from datetime import datetime, timezone

    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        return jsonify({"error": "Model not yet loaded"}), 503

    scores = [r.risk_score for r in cache.get_all()]
    n = len(scores)
    if n == 0:
        return jsonify({"error": "No predictions in cache"}), 503

    scores_sorted = sorted(scores)

    def percentile(data: list[float], p: float) -> float:
        idx = (len(data) - 1) * p / 100
        lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
        return data[lo] + (data[hi] - data[lo]) * (idx - lo)

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance)

    # 10 buckets: [0.0,0.1), [0.1,0.2), ..., [0.9,1.0]
    BUCKETS = 10
    counts = [0] * BUCKETS
    for s in scores:
        idx = min(int(s * BUCKETS), BUCKETS - 1)
        counts[idx] += 1

    histogram = [
        {
            "bucket": f"{i / BUCKETS:.1f}–{(i + 1) / BUCKETS:.1f}",
            "from": round(i / BUCKETS, 1),
            "to": round((i + 1) / BUCKETS, 1),
            "count": counts[i],
        }
        for i in range(BUCKETS)
    ]

    high_thresh = float(os.getenv("RISK_HIGH", "0.7"))
    med_thresh = float(os.getenv("RISK_MEDIUM", "0.3"))
    tier_counts = {
        "high": sum(1 for s in scores if s >= high_thresh),
        "medium": sum(1 for s in scores if med_thresh <= s < high_thresh),
        "low": sum(1 for s in scores if s < med_thresh),
    }

    return jsonify({
        "total_students": n,
        "score_mean": round(mean, 4),
        "score_std": round(std, 4),
        "score_p10": round(percentile(scores_sorted, 10), 4),
        "score_p25": round(percentile(scores_sorted, 25), 4),
        "score_p50": round(percentile(scores_sorted, 50), 4),
        "score_p75": round(percentile(scores_sorted, 75), 4),
        "score_p90": round(percentile(scores_sorted, 90), 4),
        "tier_counts": tier_counts,
        "histogram": histogram,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }), 200


@routes_bp.get("/api/model")
def get_model_info():
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Find the @prod alias version (alias-based promotion, not legacy stages)
        model_name = os.getenv("MLFLOW_MODEL_NAME", "nextstep-lstm")
        try:
            target = client.get_model_version_by_alias(model_name, "prod")
        except Exception:
            # Fallback: latest registered version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return jsonify({"error": "No registered model found"}), 404
            target = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

        run = client.get_run(target.run_id)

        # Parse log-model history for size & flavors
        import json as _json

        history = _json.loads(run.data.tags.get("mlflow.log-model.history", "[]"))
        artifact = history[0] if history else {}
        flavors = artifact.get("flavors", {})
        pytorch_flavor = flavors.get("pytorch", {})

        # Normalize metric names: handle both old (train_lstm.py) and new (registry.py) naming
        _METRIC_ALIASES = {
            "test_auc": "val_auc",  # old name → canonical
            "test_f1": "val_f1",
            "val_f1_optimal": "val_f1_internal",
        }
        raw_metrics = dict(run.data.metrics)
        metrics: dict = {}
        for k, v in raw_metrics.items():
            canonical = _METRIC_ALIASES.get(k, k)
            metrics[canonical] = v

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
                "metrics": metrics,
            }
        ), 200
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to fetch model info: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
