"""
LLMService — builds LGPD-safe prompts and calls Groq API.

Single Responsibility: generate pedagogical advice text.

P0: system/user role split, better model, temperature, max_tokens, output format
P1: in-memory response cache (hash-based), retry with backoff, response validation,
    enriched prompt context (worst indicator, risk tier label)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

log = logging.getLogger(__name__)

FALLBACK_TEXT = (
    "A sugestão pedagógica não pôde ser gerada no momento. "
    "Por favor, consulte o histórico de indicadores do estudante e entre em contato "
    "com o coordenador pedagógico para orientações personalizadas."
)

# ── Model config ──────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"  # Groq free tier — much better than 8b
MAX_TOKENS = 500  # enough for 4 well-formed bullet points
TEMPERATURE = 0.5  # grounded but not robotic
TIMEOUT = 10.0  # 70b is still fast on Groq (~1-2s)
MAX_RETRIES = 1  # 1 retry before fallback
RETRY_DELAY = 1.0  # seconds between retries

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_TTL_SECONDS = 3600  # 1 hour

_INDICATOR_LABELS: dict[str, str] = {
    "iaa": "Desempenho Acadêmico (IAA)",
    "ieg": "Engajamento (IEG)",
    "ips": "Índice Psicossocial (IPS)",
    "ida": "Aprendizagem (IDA)",
    "ipv": "Visão de Vida (IPV)",
    "ipp": "Índice Psicopedagógico (IPP)",
    "ian": "Adequação ao Nível (IAN)",
    "inde": "Desenvolvimento Educacional (INDE)",
    "defasagem": "Defasagem Escolar (anos atrás)",
}

# Thresholds considered "low" per indicator (below = weak signal)
_WEAK_THRESHOLD: dict[str, float] = {
    "iaa": 5.5,
    "ieg": 5.5,
    "ips": 5.5,
    "ida": 5.5,
    "ipv": 5.5,
    "ipp": 5.5,
    "ian": 5.5,
    "inde": 5.5,
}

_SYSTEM_PROMPT = """\
Você é um assistente pedagógico especializado em análise de risco de defasagem escolar, \
apoiando coordenadores da ONG Passos Mágicos.

Regras obrigatórias:
1. Nunca mencione nome completo, RA, CPF, turma, endereço ou qualquer dado pessoal identificável.
2. Responda APENAS em português brasileiro, de forma direta e acionável.
3. Formate a resposta como uma lista numerada com exatamente 4 sugestões pedagógicas.
4. Cada sugestão deve ter no máximo 2 frases — objetiva e prática.
5. Não invente dados que não estejam no contexto fornecido.
6. Não inclua introdução, conclusão ou saudações — apenas as 4 sugestões.\
"""


class LLMService:
    """
    Generates pedagogical advice via Groq.
    Never raises — always returns (text, is_fallback).
    """

    def __init__(self, api_key: str | None = None, model: str = MODEL) -> None:
        self._api_key = api_key or os.getenv("GROQ_API_KEY")
        self._model = model
        self._cache: dict[str, tuple[str, float]] = {}  # key → (text, expires_at)
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
        """Return (advice_text, is_fallback). Never raises."""
        cache_key = self._cache_key(display_name, indicators, risk_score)
        cached = self._get_cache(cache_key)
        if cached:
            log.info("LLM cache hit for %s", display_name)
            return cached, False

        for attempt in range(MAX_RETRIES + 1):
            try:
                text = self._call_api(display_name, indicators, risk_score)
                validated = self._validate(text)
                self._set_cache(cache_key, validated)
                return validated, False
            except Exception as exc:  # noqa: BLE001
                if attempt < MAX_RETRIES:
                    log.warning("LLM attempt %d failed (%s) — retrying in %.1fs", attempt + 1, exc, RETRY_DELAY)
                    time.sleep(RETRY_DELAY)
                else:
                    log.warning("LLM all attempts failed (%s) — returning fallback", exc)

        return FALLBACK_TEXT, True

    # ── Private — API call ────────────────────────────────────────────────────

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
        user_prompt = self._build_user_prompt(display_name, indicators, risk_score)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    # ── Private — prompt ──────────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(
        display_name: str,
        indicators: dict[str, Any],
        risk_score: float,
    ) -> str:
        """LGPD-safe prompt with enriched context."""
        risk_pct = f"{risk_score * 100:.1f}%"
        risk_label = "ALTO" if risk_score >= 0.65 else "MÉDIO" if risk_score >= 0.35 else "BAIXO"

        # Build indicator lines + identify weakest (sorted worst-first)
        lines: list[str] = []
        weak: list[tuple[float, str]] = []  # (sort_value, label) — lower = worse
        for key, label in _INDICATOR_LABELS.items():
            val = indicators.get(key)
            if val is None:
                formatted = "dado não disponível"
            elif key == "defasagem":
                v = int(val)
                formatted = f"{v:+d} {'fase' if abs(v) == 1 else 'fases'}"
                if v < 0:
                    weak.append((float(v), label))
            else:
                fv = float(val)
                formatted = f"{fv:.2f}"
                # IEG=0 or IDA=0 in the display data means the original record
                # was zero — very likely a data-entry error or student with no
                # evaluation history, NOT a genuine score of zero.
                if key in ("ieg", "ida") and fv == 0.0:
                    formatted += (
                        " ⚠️ (provável erro de registro ou aluno sem histórico — não interpretar como desempenho nulo)"
                    )
                if key in _WEAK_THRESHOLD and fv < _WEAK_THRESHOLD[key]:
                    weak.append((fv, label))
            lines.append(f"  • {label}: {formatted}")

        # Sort worst → best and take up to 4 (one per suggestion)
        weak_sorted = [label for _, label in sorted(weak, key=lambda x: x[0])]

        indicators_block = "\n".join(lines)
        weak_block = (
            f"\nIndicadores mais preocupantes (priorizar nesta ordem): {', '.join(weak_sorted[:4])}."
            if weak_sorted
            else ""
        )

        # Urgency notice for students already behind
        defasagem_val = indicators.get("defasagem")
        if defasagem_val is not None and int(defasagem_val) < 0:
            fases = abs(int(defasagem_val))
            fase_word = "fase" if fases == 1 else "fases"
            urgency_block = (
                f"\n⚠️ ATENÇÃO: Este aluno já está {fases} {fase_word} abaixo do esperado para sua idade. "
                f"Priorize intervenções de recuperação imediata e acompanhamento próximo. "
                f"Ao menos uma sugestão deve abordar diretamente a redução da defasagem escolar acumulada."
            )
        else:
            urgency_block = ""

        return (
            f"Aluno(a): {display_name}\n"
            f"Risco de defasagem no próximo ciclo: {risk_pct} (nível {risk_label})"
            f"{urgency_block}\n\n"
            f"Indicadores atuais:\n{indicators_block}\n"
            f"{weak_block}\n\n"
            f"Forneça 4 sugestões pedagógicas numeradas, objetivas e práticas, "
            f"que o coordenador pode adotar imediatamente para apoiar este aluno."
        )

    # ── Private — validation ──────────────────────────────────────────────────

    @staticmethod
    def _validate(text: str) -> str:
        """Raise if response looks empty or truncated; strip otherwise."""
        stripped = text.strip()
        if not stripped:
            raise ValueError("LLM returned empty response")
        if len(stripped) < 80:
            raise ValueError(f"LLM response suspiciously short ({len(stripped)} chars)")
        return stripped

    # ── Private — cache ───────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(display_name: str, indicators: dict[str, Any], risk_score: float) -> str:
        payload = json.dumps(
            {"n": display_name, "i": indicators, "r": round(risk_score, 3)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_cache(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if entry and time.monotonic() < entry[1]:
            return entry[0]
        self._cache.pop(key, None)
        return None

    def _set_cache(self, key: str, text: str) -> None:
        self._cache[key] = (text, time.monotonic() + _CACHE_TTL_SECONDS)
