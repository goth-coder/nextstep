"""
LLMService — builds LGPD-safe prompts and calls Groq API.

Single Responsibility: generate pedagogical advice text.
Extracted from the old llm_service.py into a class for testability.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

FALLBACK_TEXT = (
    "A sugestão pedagógica não pôde ser gerada no momento. "
    "Por favor, consulte o histórico de indicadores do estudante e entre em contato "
    "com o coordenador pedagógico para orientações personalizadas."
)

MODEL = "llama3-8b-8192"
MAX_TOKENS = 250
TIMEOUT = 5.0

_INDICATOR_LABELS: dict[str, str] = {
    "iaa": "Desempenho Acadêmico (IAA)",
    "ieg": "Engajamento (IEG)",
    "ips": "Índice Psicossocial (IPS)",
    "ida": "Autossuficiência (IDA)",
    "ipv": "Visão de Vida (IPV)",
    "ipp": "Índice Psicopedagógico (IPP)",
    "inde": "Desenvolvimento Educacional (INDE)",
    "defasagem": "Defasagem Escolar Atual",
}


class LLMService:
    """Generates pedagogical advice via Groq. Never raises — always returns fallback on error."""

    def __init__(self, api_key: str | None = None, model: str = MODEL) -> None:
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._model = model
        try:
            import groq as groq_sdk

            self._groq = groq_sdk
        except ImportError:
            self._groq = None  # type: ignore[assignment]

    # ── Public ────────────────────────────────────────────────────────────────

    def generate_advice(
        self,
        display_name: str,
        indicators: dict[str, Any],
        risk_score: float,
    ) -> tuple[str, bool]:
        """
        Return (advice_text, is_fallback).
        Never raises — returns fallback text on any failure.
        """
        try:
            return self._call_api(display_name, indicators, risk_score), False
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM call failed (%s) — returning fallback", exc)
            return FALLBACK_TEXT, True

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_api(
        self,
        display_name: str,
        indicators: dict[str, Any],
        risk_score: float,
    ) -> str:
        if self._groq is None:
            raise RuntimeError("groq package not installed")
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        client = self._groq.Groq(api_key=self._api_key, timeout=TIMEOUT)
        prompt = self._build_prompt(display_name, indicators, risk_score)
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _build_prompt(
        display_name: str,
        indicators: dict[str, Any],
        risk_score: float,
    ) -> str:
        """LGPD-safe prompt — no RA, CPF, full name, or class_group."""
        risk_pct = f"{risk_score * 100:.1f}%"
        lines = []
        for key, label in _INDICATOR_LABELS.items():
            val = indicators.get(key)
            if val is None:
                formatted = "dado não disponível"
            elif key == "defasagem":
                formatted = f"{int(val):+d} fases"
            elif key == "fase_num":
                continue
            else:
                formatted = f"{float(val):.2f}"
            lines.append(f"  - {label}: {formatted}")

        indicators_text = "\n".join(lines)
        return (
            f"Você é um assistente pedagógico especializado em análise de risco de defasagem escolar.\n\n"
            f"O estudante identificado como '{display_name}' apresenta um risco de {risk_pct} "
            f"de continuar em defasagem escolar no próximo ciclo.\n\n"
            f"Indicadores atuais:\n{indicators_text}\n\n"
            f"Com base nesses dados, forneça 3 a 4 sugestões pedagógicas objetivas e práticas "
            f"que o coordenador pedagógico pode adotar para apoiar este estudante. "
            f"Responda em português, de forma clara e acionável. "
            f"Não inclua qualquer dado pessoal identificável na resposta."
        )
