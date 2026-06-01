"""Swagger / OpenAPI configuration for the NextStep API."""

from flasgger import Swagger

SWAGGER_TEMPLATE = {
    "info": {
        "title": "NextStep API",
        "description": (
            "Educational lag risk prediction API for the NGO Passos Mágicos.\n\n"
            "The model predicts P(lag worsening) for the next academic cycle, "
            "using pedagogical indicators and students' academic data."
        ),
        "version": "1.0.0",
    },
    "basePath": "/",
    "schemes": ["http", "https"],
}

SWAGGER_CONFIG = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}


def init_swagger(app):
    """Attach Swagger UI to the Flask app."""
    Swagger(app, template=SWAGGER_TEMPLATE, config=SWAGGER_CONFIG)
