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
    "The pedagogical suggestion could not be generated at this time. "
    "Please review the student's indicator history and contact "
    "the pedagogical coordinator for personalized guidance."
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
    "iaa": "Academic Achievement Index (IAA)",
    "ieg": "Engagement Index (IEG)",
    "ips": "Psychosocial Index (IPS)",
    "ida": "Learning Index (IDA)",
    "ipv": "Future Vision Index (IPV)",
    "ipp": "Psychopedagogical Index (IPP)",
    "ian": "Grade Level Adequacy Index (IAN)",
    "inde": "Educational Development Index (INDE)",
    "defasagem": "Educational Lag (years behind)",
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
You are a pedagogical assistant specialized in educational lag risk analysis, \
supporting coordinators at the NGO Passos Mágicos.

Mandatory rules:
1. Never mention full names, student IDs, tax IDs, class groups, addresses, or any personally identifiable data.
2. Respond ONLY in English, in a direct and actionable manner.
3. Format the response as a numbered list with exactly 4 pedagogical suggestions.
4. Each suggestion must be at most 2 sentences — objective and practical.
5. Do not invent data that is not present in the provided context.
6. Do not include introductions, conclusions, or greetings — only the 4 suggestions.
7. Indicators marked with ⚠️ are missing or invalid data — IGNORE THEM completely. \
Never suggest improving records, fixing registrations, or filling in data. Focus only on pedagogical actions.\
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
        risk_label = "HIGH" if risk_score >= 0.65 else "MEDIUM" if risk_score >= 0.35 else "LOW"

        # Build indicator lines + identify weakest (sorted worst-first)
        lines: list[str] = []
        weak: list[tuple[float, str]] = []  # (sort_value, label) — lower = worse
        for key, label in _INDICATOR_LABELS.items():
            val = indicators.get(key)
            if val is None:
                formatted = "data not available"
            elif key == "defasagem":
                v = int(val)
                formatted = f"{v:+d} {'phase' if abs(v) == 1 else 'phases'}"
                if v < 0:
                    weak.append((float(v), label))
            else:
                fv = float(val)
                formatted = f"{fv:.2f}"
                # Zero in the display data very likely means data-entry error
                # or student with no evaluation — NOT a genuine score.
                if key in _WEAK_THRESHOLD and fv == 0.0:
                    formatted += " ⚠️ (data missing — ignore)"
                elif key in _WEAK_THRESHOLD and fv < _WEAK_THRESHOLD[key]:
                    weak.append((fv, label))
            lines.append(f"  • {label}: {formatted}")

        # Sort worst → best and take up to 4 (one per suggestion)
        weak_sorted = [label for _, label in sorted(weak, key=lambda x: x[0])]

        indicators_block = "\n".join(lines)
        weak_block = (
            f"\nMost concerning indicators (prioritize in this order): {', '.join(weak_sorted[:4])}."
            if weak_sorted
            else ""
        )

        # Urgency notice for students already behind
        defasagem_val = indicators.get("defasagem")
        if defasagem_val is not None and int(defasagem_val) < 0:
            fases = abs(int(defasagem_val))
            phase_word = "phase" if fases == 1 else "phases"
            urgency_block = (
                f"\n⚠️ WARNING: This student is already {fases} {phase_word} below the expected level for their age. "
                f"Prioritize immediate recovery interventions and close monitoring. "
                f"At least one suggestion must directly address reducing the accumulated educational lag."
            )
        else:
            urgency_block = ""

        return (
            f"Student: {display_name}\n"
            f"Educational lag risk in the next cycle: {risk_pct} (level {risk_label})"
            f"{urgency_block}\n\n"
            f"Current indicators:\n{indicators_block}\n"
            f"{weak_block}\n\n"
            f"Provide 4 numbered, objective, and practical pedagogical suggestions "
            f"that the coordinator can adopt immediately to support this student."
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
