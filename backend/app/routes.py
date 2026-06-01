"""
Flask Blueprint: REST API routes.

Endpoints:
    GET  /health
    GET  /readyz
    GET  /api/students
    GET  /api/students/<int:student_id>
    GET  /api/students/<int:student_id>/advice
    POST /api/predict
    POST /api/predict/batch
    GET  /api/model/drift
    GET  /api/model
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import mlflow
from flask import Blueprint, current_app, jsonify, request

from app.limiter import limiter

log = logging.getLogger(__name__)
routes_bp = Blueprint("routes", __name__)


@routes_bp.get("/health")
def health():
    """Health check.
    ---
    tags:
      - System
    responses:
      200:
        description: API health status
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            model_loaded:
              type: boolean
            student_count:
              type: integer
    """
    cache = current_app.extensions["cache"]
    payload = {
        "status": "ok",
        "model_loaded": cache.is_ready(),
        "student_count": cache.count(),
        "cache_attempts": cache.attempts(),
        "last_cache_attempt_at": cache.last_attempt_at(),
    }
    if not cache.is_ready() and cache.last_error():
        payload["cache_error"] = cache.last_error()
    return jsonify(payload), 200


@routes_bp.get("/readyz")
def readyz():
    """Readiness probe.
    ---
    tags:
      - System
    responses:
      200:
        description: API ready
        schema:
          type: object
          properties:
            status:
              type: string
              example: ready
            model_loaded:
              type: boolean
            student_count:
              type: integer
      503:
        description: Not ready yet
    """
    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        detail = cache.last_error() or "Model not yet loaded"
        payload = {
            "status": "not-ready",
            "model_loaded": False,
            "student_count": cache.count(),
            "cache_attempts": cache.attempts(),
            "last_cache_attempt_at": cache.last_attempt_at(),
            "cache_error": detail,
        }
        return jsonify(payload), 503

    return jsonify({"status": "ready", "model_loaded": True, "student_count": cache.count()}), 200


@routes_bp.get("/api/students")
def list_students():
    """Lists all students with risk scores.
    ---
    tags:
      - Students
    responses:
      200:
        description: List of students sorted by risk
        schema:
          type: object
          properties:
            students:
              type: array
              items:
                type: object
                properties:
                  student_id:
                    type: integer
                  display_name:
                    type: string
                  phase:
                    type: string
                  class_group:
                    type: string
                  risk_score:
                    type: number
                    nullable: true
                  risk_tier:
                    type: string
                    enum: [high, medium, low]
            total:
              type: integer
      503:
        description: Data not yet loaded
    """
    cache = current_app.extensions["cache"]
    if not cache.has_students():
        detail = cache.last_error() or "Student data not yet available"
        return jsonify({"error": "Student data not yet loaded", "detail": detail}), 503
    records = cache.get_all()  # pre-sorted by risk desc at load time
    students = [
        {
            "student_id": r.student_id,
            "display_name": r.display_name,
            "phase": r.phase,
            "class_group": r.class_group,
            "risk_score": r.risk_score,
            "risk_tier": r.risk_tier.value if r.risk_tier is not None else None,
        }
        for r in records
    ]
    return jsonify({"students": students, "total": len(students)}), 200


@routes_bp.get("/api/students/<int:student_id>")
def get_student(student_id: int):
    """Student details with indicators.
    ---
    tags:
      - Students
    parameters:
      - name: student_id
        in: path
        type: integer
        required: true
        description: Student ID
    responses:
      200:
        description: Student details
        schema:
          type: object
          properties:
            student_id:
              type: integer
            display_name:
              type: string
            phase:
              type: string
            fase_num:
              type: integer
            class_group:
              type: string
            gender:
              type: integer
              description: "0=Female, 1=Male"
            age:
              type: integer
            risk_score:
              type: number
              nullable: true
            risk_tier:
              type: string
              enum: [high, medium, low]
            indicators:
              type: object
              properties:
                iaa:
                  type: number
                ieg:
                  type: number
                ips:
                  type: number
                ida:
                  type: number
                ipv:
                  type: number
                ipp:
                  type: number
                ian:
                  type: number
                inde:
                  type: number
                defasagem:
                  type: integer
                mat:
                  type: number
                por:
                  type: number
                tenure:
                  type: integer
                n_av:
                  type: integer
      404:
        description: Student not found
      503:
        description: Data not yet loaded
    """
    cache = current_app.extensions["cache"]
    if not cache.has_students():
        detail = cache.last_error() or "Student data not yet available"
        return jsonify({"error": "Student data not yet loaded", "detail": detail}), 503
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
            "risk_tier": record.risk_tier.value if record.risk_tier is not None else None,
            "indicators": {
                "iaa": ind.iaa,
                "ieg": ind.ieg,
                "ips": ind.ips,
                "ida": ind.ida,
                "ipv": ind.ipv,
                "ipp": ind.ipp,
                "ian": ind.ian,
                "inde": ind.inde,
                "defasagem": ind.defasagem,
                "mat": ind.mat,
                "por": ind.por,
                "tenure": ind.tenure,
                "n_av": ind.n_av,
            },
        }
    ), 200


@routes_bp.get("/api/students/<int:student_id>/advice")
@limiter.limit("20 per hour")
def get_advice(student_id: int):
    """LLM-generated pedagogical suggestions for a student.
    ---
    tags:
      - Students
    parameters:
      - name: student_id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Pedagogical suggestions
        schema:
          type: object
          properties:
            student_id:
              type: integer
            advice:
              type: string
              description: Text with 4 pedagogical suggestions
            is_fallback:
              type: boolean
              description: true if the LLM failed and returned the default fallback text
            generated_at:
              type: string
              format: date-time
              nullable: true
      404:
        description: Student not found
      503:
        description: Model not yet loaded
    """
    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        detail = cache.last_error() or "Model not yet loaded"
        return jsonify({"error": "Model not yet loaded", "detail": detail}), 503
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
            "ian": ind.ian,
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


@routes_bp.post("/api/predict")
@limiter.limit("60 per hour")
def predict_one():
    """On-demand risk prediction for arbitrary indicator values.
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            iaa:
              type: number
              description: "Academic Achievement Index (0-10)"
            ieg:
              type: number
              description: "Engagement Index (0-10)"
            ips:
              type: number
              description: "Psychosocial Index (0-10)"
            ida:
              type: number
              description: "Learning Index (0-10)"
            ipv:
              type: number
              description: "Future Vision Index (0-10)"
            ian:
              type: number
              description: "Grade Level Adequacy Index (0-10)"
            inde:
              type: number
              description: "Educational Development Index (0-10)"
            defasagem:
              type: integer
              description: "Educational lag (negative = behind)"
            fase_num:
              type: integer
              description: "Phase (0-9)"
            gender:
              type: integer
              description: "0=Female, 1=Male"
            age:
              type: number
            mat:
              type: number
              description: "Math grade (0-10)"
            por:
              type: number
              description: "Portuguese grade (0-10)"
            tenure:
              type: integer
              description: "Years at the NGO"
            n_av:
              type: integer
              description: "Number of evaluators"
            missing_grades:
              type: number
              description: "1 if grades were imputed, 0 otherwise"
    responses:
      200:
        description: Risk score
        schema:
          type: object
          properties:
            risk_score:
              type: number
              example: 0.7234
            risk_tier:
              type: string
              enum: [high, medium, low]
            input:
              type: object
      422:
        description: Unknown fields
      503:
        description: Model not yet loaded
    """
    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        return jsonify({"error": "Model not yet loaded"}), 503

    body = request.get_json(silent=True) or {}

    VALID_KEYS = {
        "iaa", "ieg", "ips", "ida", "ipv", "ian", "inde",
        "defasagem", "fase_num", "gender", "age", "mat", "por",
        "tenure", "n_av", "missing_grades",
    }
    unknown = set(body.keys()) - VALID_KEYS
    if unknown:
        return jsonify({"error": f"Unknown fields: {sorted(unknown)}"}), 422

    try:
        score = cache.predict_one(body)
    except Exception as exc:  # noqa: BLE001
        log.error("predict_one failed: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500

    high_thresh = float(os.getenv("RISK_HIGH", "0.7"))
    med_thresh = float(os.getenv("RISK_MEDIUM", "0.3"))
    tier = "high" if score >= high_thresh else "medium" if score >= med_thresh else "low"

    return jsonify({"risk_score": round(score, 4), "risk_tier": tier, "input": body}), 200


@routes_bp.post("/api/predict/batch")
@limiter.limit("10 per minute")
def predict_batch():
    """Batch risk prediction.
    ---
    tags:
      - Prediction
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - students
          properties:
            students:
              type: array
              items:
                type: object
                properties:
                  student_id:
                    type: integer
                  iaa:
                    type: number
                  ieg:
                    type: number
                  ips:
                    type: number
                  ida:
                    type: number
                  ipv:
                    type: number
                  ian:
                    type: number
                  inde:
                    type: number
                  defasagem:
                    type: integer
                  fase_num:
                    type: integer
                  gender:
                    type: integer
                  age:
                    type: number
                  mat:
                    type: number
                  por:
                    type: number
                  tenure:
                    type: integer
                  n_av:
                    type: integer
                  missing_grades:
                    type: number
    responses:
      200:
        description: List of scores
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  student_id:
                    type: integer
                  risk_score:
                    type: number
                  risk_tier:
                    type: string
                    enum: [high, medium, low]
      422:
        description: Invalid format
      503:
        description: Model not yet loaded
    """
    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        return jsonify({"error": "Model not yet loaded"}), 503

    body = request.get_json(silent=True) or {}
    items = body.get("students")
    if not isinstance(items, list):
        return jsonify({"error": "'students' must be a list"}), 422

    VALID_KEYS = {
        "student_id", "iaa", "ieg", "ips", "ida", "ipv", "ian", "inde",
        "defasagem", "fase_num", "gender", "age", "mat", "por",
        "tenure", "n_av", "missing_grades",
    }
    high_thresh = float(os.getenv("RISK_HIGH", "0.7"))
    med_thresh = float(os.getenv("RISK_MEDIUM", "0.3"))

    results = []
    for item in items:
        unknown = set(item.keys()) - VALID_KEYS
        if unknown:
            return jsonify({"error": f"Unknown fields in batch item: {sorted(unknown)}"}), 422
        student_id = item.get("student_id")
        features = {k: v for k, v in item.items() if k != "student_id"}
        try:
            score = cache.predict_one(features)
        except Exception as exc:  # noqa: BLE001
            log.error("predict_batch item %s failed: %s", student_id, exc, exc_info=True)
            return jsonify({"error": str(exc)}), 500
        tier = "high" if score >= high_thresh else "medium" if score >= med_thresh else "low"
        results.append({"student_id": student_id, "risk_score": round(score, 4), "risk_tier": tier})

    return jsonify({"results": results}), 200


@routes_bp.get("/api/model/drift")
def get_model_drift():
    """Score distribution statistics for drift monitoring.
    ---
    tags:
      - Model
    responses:
      200:
        description: Distribution statistics
        schema:
          type: object
          properties:
            total_students:
              type: integer
            score_mean:
              type: number
            score_std:
              type: number
            score_p10:
              type: number
            score_p25:
              type: number
            score_p50:
              type: number
            score_p75:
              type: number
            score_p90:
              type: number
            tier_counts:
              type: object
              properties:
                high:
                  type: integer
                medium:
                  type: integer
                low:
                  type: integer
            histogram:
              type: array
              items:
                type: object
                properties:
                  bucket:
                    type: string
                  count:
                    type: integer
      503:
        description: Model not yet loaded
    """
    import math
    from datetime import datetime, timezone

    cache = current_app.extensions["cache"]
    if not cache.is_ready():
        detail = cache.last_error() or "Model not yet loaded"
        return jsonify({"error": "Model not yet loaded", "detail": detail}), 503

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

    return jsonify(
        {
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
        }
    ), 200


@routes_bp.get("/api/model")
def get_model_info():
    """Registered model information from MLflow.
    ---
    tags:
      - Model
    responses:
      200:
        description: Model metadata
        schema:
          type: object
          properties:
            model_name:
              type: string
            version:
              type: string
            stage:
              type: string
            run_id:
              type: string
            run_name:
              type: string
            params:
              type: object
            metrics:
              type: object
      404:
        description: No registered model found
      500:
        description: Error fetching model information
    """
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        max_attempts = max(1, int(os.getenv("MODEL_INFO_MAX_ATTEMPTS", "4")))
        base_delay = max(0.0, float(os.getenv("MODEL_INFO_RETRY_BASE_SECONDS", "1")))

        client = None
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                mlflow.set_tracking_uri(tracking_uri)
                client = mlflow.tracking.MlflowClient()
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning("MLflow client init failed (%d/%d): %s", attempt, max_attempts, exc)
                if attempt < max_attempts:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
        if client is None and last_error is not None:
            raise last_error

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
