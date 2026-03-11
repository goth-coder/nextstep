"""Swagger / OpenAPI configuration for the NextStep API."""

from flasgger import Swagger


SWAGGER_TEMPLATE = {
    "info": {
        "title": "NextStep API",
        "description": (
            "API de predição de risco de defasagem escolar para a ONG Passos Mágicos.\n\n"
            "O modelo prediz P(piora na defasagem) para o próximo ciclo letivo, "
            "usando indicadores pedagógicos e dados acadêmicos dos alunos."
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
