"""
Domain entities.

StudentRecord  — immutable value-object representing one student + their risk prediction.
RiskTier       — categorical risk classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskTier(str, Enum):
    """Categorical risk tier derived from the model's continuous risk score."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float, high: float = 0.7, medium: float = 0.3) -> "RiskTier":
        if score >= high:
            return cls.HIGH
        if score >= medium:
            return cls.MEDIUM
        return cls.LOW


@dataclass(frozen=True)
class Indicators:
    """Raw (scaled, 0-1) PEDE indicator values for a student in a given year."""

    iaa: Optional[float] = None  # Índice de Aproveitamento Acadêmico
    ieg: Optional[float] = None  # Índice de Engajamento
    ips: Optional[float] = None  # Índice Psicossocial
    ida: Optional[float] = None  # Índice de Aprendizagem
    ipv: Optional[float] = None  # Índice de Ponto de Virada
    ipp: Optional[float] = None  # Índice Psicopedagógico
    ian: Optional[float] = None  # Índice de Adequação ao Nível
    inde: Optional[float] = None  # Índice de Desenvolvimento Educacional (composite)
    defasagem: Optional[float] = None  # Defasagem escolar no ano atual (raw int, não scaled)
    fase_num: Optional[int] = None  # Fase normalizada (0=ALFA .. 8=universitário)
    mat: Optional[float] = None  # Nota de Matemática (0-10)
    por: Optional[float] = None  # Nota de Português (0-10)
    tenure: Optional[int] = None  # Anos na ONG (year - ano_ingresso)
    n_av: Optional[int] = None  # Número de avaliadores

    def to_dict(self) -> dict:
        return {
            "iaa": self.iaa,
            "ieg": self.ieg,
            "ips": self.ips,
            "ida": self.ida,
            "ipv": self.ipv,
            "ipp": self.ipp,
            "ian": self.ian,
            "inde": self.inde,
            "defasagem": self.defasagem,
            "fase_num": self.fase_num,
            "mat": self.mat,
            "por": self.por,
            "tenure": self.tenure,
            "n_av": self.n_av,
        }


@dataclass(frozen=True)
class StudentRecord:
    """
    Immutable value-object for one student with their latest risk prediction.

    Produced by PredictionService and stored in StudentCacheService.
    """

    student_id: int
    ra: str  # Anonymised RA (e.g. "RA-42")
    display_name: str  # First token of anonymised name
    phase: str  # Human-readable phase label ("Fase 3", "ALFA", …)
    phase_num: int  # Normalised phase integer 0-8
    class_group: str  # Turma string
    gender: int  # 0=Feminino, 1=Masculino
    age: Optional[int]  # Age in the observation year
    year: int  # Year of the observation used for inference
    risk_score: Optional[float]  # Model output — None until model is trained and loaded
    risk_tier: Optional[RiskTier]  # None until model loaded
    indicators: Indicators  # Scaled indicator values

    @classmethod
    def build(
        cls,
        student_id: int,
        ra: str,
        display_name: str,
        phase: str,
        phase_num: int,
        class_group: str,
        gender: int,
        age: Optional[int],
        year: int,
        indicators: Indicators,
        risk_score: Optional[float] = None,
    ) -> "StudentRecord":
        return cls(
            student_id=student_id,
            ra=ra,
            display_name=display_name,
            phase=phase,
            phase_num=phase_num,
            class_group=class_group,
            gender=gender,
            age=age,
            year=year,
            risk_score=round(risk_score, 4) if risk_score is not None else None,
            risk_tier=RiskTier.from_score(risk_score) if risk_score is not None else None,
            indicators=indicators,
        )
