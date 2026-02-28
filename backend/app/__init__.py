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

log = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    @app.after_request
    def _add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        return response

    from app.repositories.mlflow_model import MLflowModelRepository
    from app.repositories.student_data import DiskStudentDataRepository
    from app.services.cache import StudentCacheService
    from app.services.llm import LLMService
    from app.services.prediction import PredictionService

    cache_svc = StudentCacheService(
        PredictionService(
            model_repo=MLflowModelRepository(),
            data_repo=DiskStudentDataRepository(),
        )
    )
    llm_svc = LLMService(api_key=os.getenv("GROQ_API_KEY"))

    app.extensions["cache"] = cache_svc
    app.extensions["llm"] = llm_svc

    with app.app_context():
        try:
            cache_svc.load()
        except Exception as exc:  # noqa: BLE001
            log.error("Prediction cache failed to load: %s", exc, exc_info=True)

    from app.routes import routes_bp
    app.register_blueprint(routes_bp)

    log.info("Flask app created checkmark  routes=%s", [str(r) for r in app.url_map.iter_rules()])
    return app
