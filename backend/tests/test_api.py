"""
Unit tests for Flask API routes.

Tests inject a MagicMock StudentCacheService and LLMService into
app.extensions so no MLflow, PyTorch, or Groq calls occur.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.domain.student import Indicators, RiskTier, StudentRecord

_IND_HIGH = Indicators(iaa=6.2, ieg=5.5, ips=4.3, ida=3.7, ipv=3.1, ipp=4.0, inde=4.5, defasagem=-2)
_IND_LOW  = Indicators(iaa=9.0, ieg=8.8, ips=7.7, ida=9.1, ipv=9.2, ipp=8.5, inde=9.0, defasagem=0)

REC_HIGH = StudentRecord.build(
    student_id=1, ra="RA-1", display_name="ALUNO-1",
    phase="Fase 2", phase_num=2, class_group="G", year=2024,
    gender=0, age=13,
    risk_score=0.85, indicators=_IND_HIGH,
)
REC_LOW = StudentRecord.build(
    student_id=2, ra="RA-2", display_name="ALUNO-2",
    phase="Fase 3", phase_num=3, class_group="F", year=2024,
    gender=1, age=14,
    risk_score=0.15, indicators=_IND_LOW,
)


def _make_mock_cache(ready: bool = True, records=None):
    mock_cache = MagicMock()
    mock_cache.is_ready.return_value = ready
    mock_cache.get_all.return_value = records if records is not None else [REC_HIGH, REC_LOW]
    mock_cache.get_by_id.side_effect = lambda sid: {1: REC_HIGH, 2: REC_LOW}.get(int(sid))
    mock_cache.count.return_value = len(records) if records is not None else 2
    mock_cache.attempts.return_value = 1
    mock_cache.last_attempt_at.return_value = "2026-03-04T00:00:00+00:00"
    mock_cache.last_error.return_value = None if ready else "startup load failed"
    return mock_cache


def _make_mock_llm(advice="Some advice text", fallback=False):
    mock_llm = MagicMock()
    mock_llm.generate_advice.return_value = (advice, fallback)
    return mock_llm


@pytest.fixture()
def app():
    """Flask app with mock cache and llm injected via extensions."""
    with patch("app.services.cache.StudentCacheService.load"):
        from app import create_app
        flask_app = create_app()
    flask_app.extensions["cache"] = _make_mock_cache()
    flask_app.extensions["llm"] = _make_mock_llm()
    flask_app.config["TESTING"] = True
    yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# -- GET /health ---------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["student_count"] == 2


def test_health_when_cache_not_ready():
    with patch("app.services.cache.StudentCacheService.load"):
        from app import create_app
        flask_app = create_app()
    flask_app.extensions["cache"] = _make_mock_cache(ready=False, records=[])
    flask_app.extensions["llm"] = _make_mock_llm()
    flask_app.config["TESTING"] = True
    resp = flask_app.test_client().get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["model_loaded"] is False


def test_readyz_returns_200_when_cache_ready(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True


def test_readyz_returns_503_when_cache_not_ready():
    with patch("app.services.cache.StudentCacheService.load"):
        from app import create_app
        flask_app = create_app()
    flask_app.extensions["cache"] = _make_mock_cache(ready=False, records=[])
    flask_app.extensions["llm"] = _make_mock_llm()
    flask_app.config["TESTING"] = True
    resp = flask_app.test_client().get("/readyz")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "not-ready"
    assert data["model_loaded"] is False


# -- GET /api/students ---------------------------------------------------------

def test_list_students_returns_sorted_by_risk_desc(client):
    resp = client.get("/api/students")
    assert resp.status_code == 200
    data = resp.get_json()
    scores = [s["risk_score"] for s in data["students"]]
    assert scores == sorted(scores, reverse=True), "Students must be sorted by risk_score descending"


def test_list_students_risk_tier_high(client):
    resp = client.get("/api/students")
    data = resp.get_json()
    high_student = next(s for s in data["students"] if s["student_id"] == 1)
    assert high_student["risk_tier"] == "high"


def test_list_students_risk_tier_low(client):
    resp = client.get("/api/students")
    data = resp.get_json()
    low_student = next(s for s in data["students"] if s["student_id"] == 2)
    assert low_student["risk_tier"] == "low"


def test_list_students_total_matches_count(client):
    resp = client.get("/api/students")
    data = resp.get_json()
    assert data["total"] == len(data["students"])


def test_list_students_503_when_cache_not_ready():
    with patch("app.services.cache.StudentCacheService.load"):
        from app import create_app
        flask_app = create_app()
    flask_app.extensions["cache"] = _make_mock_cache(ready=False, records=[])
    flask_app.extensions["llm"] = _make_mock_llm()
    flask_app.config["TESTING"] = True
    resp = flask_app.test_client().get("/api/students")
    assert resp.status_code == 503


# -- GET /api/students/<id> ----------------------------------------------------

def test_get_student_200(client):
    resp = client.get("/api/students/1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["student_id"] == 1
    assert "indicators" in data
    assert "iaa" in data["indicators"]


def test_get_student_404(client):
    resp = client.get("/api/students/999")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


# -- GET /api/students/<id>/advice ---------------------------------------------

def test_get_advice_always_200(client):
    resp = client.get("/api/students/1/advice")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "advice" in data
    assert "is_fallback" in data


def test_get_advice_fallback_200(app):
    app.extensions["llm"] = _make_mock_llm(advice="Fallback text", fallback=True)
    resp = app.test_client().get("/api/students/1/advice")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["is_fallback"] is True
    assert data["generated_at"] is None


def test_get_advice_404_for_missing_student(client):
    resp = client.get("/api/students/999/advice")
    assert resp.status_code == 404


def test_get_advice_503_when_cache_not_ready():
    with patch("app.services.cache.StudentCacheService.load"):
        from app import create_app
        flask_app = create_app()
    flask_app.extensions["cache"] = _make_mock_cache(ready=False, records=[])
    flask_app.extensions["llm"] = _make_mock_llm()
    flask_app.config["TESTING"] = True
    resp = flask_app.test_client().get("/api/students/1/advice")
    assert resp.status_code == 503
