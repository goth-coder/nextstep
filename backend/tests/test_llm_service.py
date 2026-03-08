"""
Unit tests for LLMService (app.services.llm).

Key concerns:
- LGPD: no PII (numeric IDs, class_group) in Groq prompt
- Fallback pattern: (text, bool) -- never raises
- Timeout/error handling: any Groq exception -> (FALLBACK_TEXT, True)
"""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock

from app.services.llm import FALLBACK_TEXT, LLMService

# -- Helpers ------------------------------------------------------------------


def _fake_groq_module(response_text: str = "Advice text", raises: Exception | None = None) -> ModuleType:
    """Build a minimal fake groq module that mimics groq.Groq().chat.completions.create()."""
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    module = ModuleType("groq")
    if raises is not None:
        module.Groq = MagicMock(
            return_value=MagicMock(chat=MagicMock(completions=MagicMock(create=MagicMock(side_effect=raises))))
        )
    else:
        module.Groq = MagicMock(
            return_value=MagicMock(chat=MagicMock(completions=MagicMock(create=MagicMock(return_value=mock_response))))
        )
    # Copy real exception types so tests can catch them
    try:
        import groq as _real_groq

        module.APITimeoutError = _real_groq.APITimeoutError
    except Exception:
        module.APITimeoutError = Exception
    return module


def _make_service(fake_groq_mod: ModuleType, api_key: str = "fake-key") -> LLMService:
    """Instantiate LLMService with a pre-built fake groq module injected directly."""
    svc = LLMService.__new__(LLMService)
    svc._api_key = api_key
    svc._model = "llama3-8b-8192"
    svc._groq = fake_groq_mod
    svc._cache = {}  # __new__ bypasses __init__ — must initialise manually
    return svc


_DEFAULT_INDICATORS = {
    "iaa": 6.2,
    "ieg": 5.5,
    "ips": 4.3,
    "ida": 3.7,
    "ipv": 3.1,
    "ipp": 4.0,
    "inde": 4.5,
    "defasagem": -2,
}


# -- LGPD compliance ----------------------------------------------------------


def test_prompt_does_not_contain_student_id():
    """generate_advice must not accept student_id -- it does not exist in the signature."""
    import inspect

    sig = inspect.signature(LLMService.generate_advice)
    assert "student_id" not in sig.parameters, "generate_advice must not accept student_id"


def test_prompt_does_not_contain_class_group():
    """generate_advice must not accept class_group -- LGPD."""
    import inspect

    sig = inspect.signature(LLMService.generate_advice)
    assert "class_group" not in sig.parameters, "generate_advice must not accept class_group"


def test_prompt_has_no_none_literal():
    """None indicator values must appear as 'dado nao disponivel', not the Python literal None."""
    captured: list[str] = []

    fake_mod = _fake_groq_module("Advice")
    orig_create = fake_mod.Groq.return_value.chat.completions.create

    def capturing_create(**kwargs):
        for m in kwargs.get("messages", []):
            captured.append(m.get("content", ""))
        return orig_create(**kwargs)

    fake_mod.Groq.return_value.chat.completions.create = capturing_create

    svc = _make_service(fake_mod)
    svc.generate_advice(
        display_name="ALUNO-42",
        indicators={k: None for k in _DEFAULT_INDICATORS},
        risk_score=0.82,
    )
    full = " ".join(captured)
    assert "None" not in full, "Python None must not appear in Groq prompt"
    # The prompt uses "dado não disponível" — check with the actual accented form
    assert "dispon" in full.lower(), "Prompt must contain a 'not available' marker for None indicators"


# -- Successful response ------------------------------------------------------


_VALID_ADVICE = (
    "1. Reforçar leitura diária com textos adequados à fase do aluno.\n"
    "2. Propor atividades de matemática contextualizada com jogos.\n"
    "3. Estabelecer metas semanais claras com feedback positivo.\n"
    "4. Envolver a família no acompanhamento das tarefas escolares."
)  # >= 80 chars — passes _validate()


def test_success_returns_advice_and_false():
    svc = _make_service(_fake_groq_module(_VALID_ADVICE))
    advice, fallback = svc.generate_advice(
        display_name="ALUNO-1",
        indicators=_DEFAULT_INDICATORS,
        risk_score=0.85,
    )
    assert fallback is False
    assert advice == _VALID_ADVICE


# -- Fallback handling --------------------------------------------------------


def test_fallback_on_timeout():
    """APITimeoutError must yield (FALLBACK_TEXT, True)."""
    try:
        import groq as _real_groq

        exc = _real_groq.APITimeoutError(request=MagicMock())
    except Exception:
        exc = RuntimeError("timeout")
    svc = _make_service(_fake_groq_module(raises=exc))
    advice, fallback = svc.generate_advice(display_name="ALUNO-1", indicators=_DEFAULT_INDICATORS, risk_score=0.85)
    assert fallback is True
    assert advice == FALLBACK_TEXT


def test_fallback_on_generic_exception():
    svc = _make_service(_fake_groq_module(raises=RuntimeError("Network error")))
    advice, fallback = svc.generate_advice(display_name="ALUNO-1", indicators=_DEFAULT_INDICATORS, risk_score=0.85)
    assert fallback is True
    assert advice == FALLBACK_TEXT


def test_fallback_when_api_key_missing():
    """Missing API key must short-circuit to fallback without calling Groq."""
    svc = _make_service(_fake_groq_module(), api_key="")
    advice, fallback = svc.generate_advice(display_name="ALUNO-1", indicators=_DEFAULT_INDICATORS, risk_score=0.85)
    assert fallback is True
    assert advice == FALLBACK_TEXT
