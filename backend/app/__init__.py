"""
Flask application factory.

Dependency Injection wiring:
    StudentCacheService  -> app.extensions["cache"]
    LLMService           -> app.extensions["llm"]

Routes receive services via current_app.extensions -- no module-level globals.
"""

import logging
import os

from flask import Flask

from app.limiter import limiter

log = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # ── CORS ─────────────────────────────────────────────────────────────────
    # In production, restrict to the deployed domain via ALLOWED_ORIGIN env var.
    # Falls back to "*" only when the var is unset (local dev).
    allowed_origin = os.getenv("ALLOWED_ORIGIN", "*")

    @app.after_request
    def _add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    # ── Rate limiting ─────────────────────────────────────────────────────────
    limiter.init_app(app)

    from app.repositories.mlflow_model import MLflowModelRepository
    from app.repositories.student_data import DiskStudentDataRepository
    from app.services.cache import StudentCacheService
    from app.services.llm import LLMService
    from app.services.prediction import PredictionService

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    log.info("Starting API with MLFLOW_TRACKING_URI=%s", tracking_uri)

    cache_svc = StudentCacheService(
        PredictionService(
            model_repo=MLflowModelRepository(tracking_uri=tracking_uri),
            data_repo=DiskStudentDataRepository(),
        )
    )
    llm_svc = LLMService(api_key=os.getenv("GROQ_API_KEY"))

    app.extensions["cache"] = cache_svc
    app.extensions["llm"] = llm_svc

    with app.app_context():
        try:
            cache_svc.load_with_retry()
        except Exception as exc:  # noqa: BLE001
            log.error("Prediction cache failed to load: %s", exc, exc_info=True)

    from app.routes import routes_bp

    app.register_blueprint(routes_bp)

    log.info("Flask app created checkmark  routes=%s", [str(r) for r in app.url_map.iter_rules()])
    return app
