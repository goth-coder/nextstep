"""
ETL pipeline for PEDE 2022-2024 dataset.

Strategy
--------
Target: P(aluno piora no próximo ciclo) — risco de aprofundamento da defasagem.

  y = 1  if  defasagem_next < defasagem_current  (defasagem piorou, ficou mais negativa)
  y = 0  if  defasagem_next >= defasagem_current (manteve ou melhorou)

Exemplos:
  def=-2, próximo=-2 → y=0  (estável)
  def=-2, próximo=-3 → y=1  (piorou ✓)
  def=-2, próximo=-1 → y=0  (melhorou)
  def= 0, próximo=-1 → y=1  (começou a defasar ✓)
  def= 1, próximo= 0 → y=0  (ainda adiantado, menos)

NOTE: o target antigo era (defasagem_next < 0) — "o aluno está defasado?"
Isso é o estado atual, não a transição. Foi corrigido para capturar a
direção do movimento: predict P(piora) dado o estado de hoje.

Training pairs (temporal, no leakage):
  • Train — year 2022 features → defasagem 2023 label  (600 pairs)
  • Test  — year 2023 features → defasagem 2024 label  (765 pairs)

Inference:
  • All 2024 students (1 156) → predict risk for the 2025 cycle.

Feature set (INPUT_SIZE = 8):
  IAA, IEG, IPS, IDA, IPV, INDE (imputed), defasagem_t (current year raw int), fase_num (0-8)

Note on defasagem feature vs target:
  - Feature: defasagem (year t) — current lag value, strong but legitimate predictor
  - Target: (defasagem_next < 0) — will the student still be lagging in year t+1?
  These are different years: no leakage, but defasagem likely dominates the model.
  Run SHAP to confirm feature importance distribution.

Display-only (stored in students_meta, NOT passed to the model):
  IPP — absent in 2022 (would be 100% synthetic for training pairs)

Dropped from model entirely:
  IAN  — leakage (corr 0.84-0.87 with target by definition)
  defasagem_bin — derived manually, redundant

Usage:
    python ml/data_loader.py
    DATA_PATH=/path/to/xlsx python ml/data_loader.py
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("data_loader")

# ── Paths ──────────────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).parent
_BACKEND_DIR = _SRC_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent

PROCESSED_DIR = _BACKEND_DIR / "data" / "processed"

_XLSX_CANDIDATES = [
    Path(os.environ.get("DATA_PATH", "")),
    _PROJECT_ROOT / "data" / "BASE DE DADOS PEDE 2022-2024 - DATATHON.xlsx",
    _BACKEND_DIR / "data" / "raw" / "BASE DE DADOS PEDE 2022-2024 - DATATHON.xlsx",
    Path("/data/BASE DE DADOS PEDE 2022-2024 - DATATHON.xlsx"),
    Path("/app/data/raw/BASE DE DADOS PEDE 2022-2024 - DATATHON.xlsx"),
]

# ── Feature constants ──────────────────────────────────────────────────────────
# IPP excluded from model features: absent in 2022 (100% synthetic if imputed for train pairs).
# IPP is still stored in students_meta for frontend display.
FEATURES = ["IAA", "IEG", "IPS", "IDA", "IPV", "INDE", "defasagem", "fase_num", "gender", "age"]
INPUT_SIZE = len(FEATURES)  # 10


# ── Demographic helpers ───────────────────────────────────────────────────────


def _encode_gender(val) -> int:
    """Encode gender to binary: Feminino/Menina → 0, Masculino/Menino → 1."""
    s = str(val).strip().lower()
    if s in ("feminino", "menina", "f"):
        return 0
    if s in ("masculino", "menino", "m"):
        return 1
    return 0  # default fallback (no nulls in dataset)


def _get_age_col(df: pd.DataFrame, year: int) -> str | None:
    """Find the age column for a given year sheet."""
    for col in [f"Idade {str(year)[-2:]}", f"Idade {year}", "Idade"]:
        if col in df.columns:
            return col
    return None


def _extract_age_value(val) -> float | None:
    """
    Extract numeric age from a raw cell value.

    In PEDE2023, the Idade column is stored as an Excel date serial misinterpreted
    as datetime: datetime(1900, 1, 8) means serial 8 → age 8.
    We recover the original integer by computing days since 1899-12-31.
    """
    import datetime as _dt

    if isinstance(val, _dt.datetime):
        # Excel epoch: serial 1 = 1900-01-01, so delta from 1899-12-31 gives serial
        epoch = _dt.datetime(1899, 12, 31)
        return float((val - epoch).days)
    try:
        v = float(val)
        return None if (v != v) else v  # NaN check
    except (TypeError, ValueError):
        return None


# ── Fase normalisation ────────────────────────────────────────────────────────


def _normalise_fase_2022(raw) -> int:
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return 0


def _normalise_fase_2023(raw) -> int:
    s = str(raw).strip().upper()
    if s == "ALFA":
        return 0
    m = re.search(r"\d+", s)
    return int(m.group()) if m else 0


def _normalise_fase_2024(raw) -> int:
    """'ALFA'→0, '1A'-'1N'→1, '2A'-'2U'→2, ..., '8A'-'8F'→8, '9'→9."""
    s = str(raw).strip().upper()
    if s == "ALFA":
        return 0
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else 0


def _find_inde_col(df: pd.DataFrame, year: int) -> str | None:
    for c in [f"INDE {year}", f"INDE {str(year)[-2:]}"]:
        if c in df.columns:
            return c
    return None


def _load_sheet(xl: pd.ExcelFile, sheet: str, year: int) -> pd.DataFrame:
    df = xl.parse(sheet)

    target_col = next((c for c in df.columns if c.strip() in ("Defas", "Defasagem")), None)
    nome_col = next((c for c in df.columns if c.strip() in ("Nome", "Nome Anonimizado")), None)
    inde_col = _find_inde_col(df, year)

    renames: dict[str, str] = {}
    if target_col:
        renames[target_col] = "defasagem_raw"
    if nome_col:
        renames[nome_col] = "Nome"
    if inde_col:
        renames[inde_col] = "INDE"

    df = df.rename(columns=renames)

    if "INDE" in df.columns:
        df["INDE"] = pd.to_numeric(df["INDE"], errors="coerce")

    df["year"] = year
    normalise_fn = {2022: _normalise_fase_2022, 2023: _normalise_fase_2023, 2024: _normalise_fase_2024}[year]
    df["fase_num"] = df["Fase"].apply(normalise_fn)
    # Rename defasagem_raw → defasagem for clarity in feature matrix
    if "defasagem_raw" in df.columns:
        df["defasagem"] = df["defasagem_raw"]

    # Demographic features
    if "Gênero" in df.columns:
        df["gender"] = df["Gênero"].apply(_encode_gender)
    else:
        df["gender"] = 0

    age_col = _get_age_col(df, year)
    if age_col:
        df["age"] = df[age_col].apply(_extract_age_value)
    else:
        df["age"] = np.nan

    return df


def _find_xlsx():
    """Return the xlsx path (local Path or gs:// URI string)."""
    data_path = os.environ.get("DATA_PATH", "")
    if data_path.startswith("gs://"):
        log.info("Using GCS data path: %s", data_path)
        return data_path  # pandas + gcsfs reads gs:// URIs natively
    for p in _XLSX_CANDIDATES:
        if p and p.is_file():
            return p
    raise FileNotFoundError("PEDE xlsx not found. Searched:\n" + "\n".join(f"  {p}" for p in _XLSX_CANDIDATES if p))


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def run_etl() -> None:
    path = _find_xlsx()
    log.info("Loading dataset from %s", path)
    xl = pd.ExcelFile(path)

    d22 = _load_sheet(xl, "PEDE2022", 2022)
    d23 = _load_sheet(xl, "PEDE2023", 2023)
    d24 = _load_sheet(xl, "PEDE2024", 2024)
    log.info("Rows: 2022=%d  2023=%d  2024=%d", len(d22), len(d23), len(d24))

    # ── 3. Temporal train/test pairs ─────────────────────────────────────────
    # IPP is NOT a model feature (absent in 2022 → would be 100% synthetic for train pairs).
    # IPP is stored separately in students_meta for frontend display only.
    FEATURE_SOURCE_COLS = ["IAA", "IEG", "IPS", "IDA", "IPV", "INDE", "defasagem", "fase_num", "gender", "age"]

    def make_pairs(df_t: pd.DataFrame, df_t1: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
        shared = set(df_t["RA"]) & set(df_t1["RA"])
        t = df_t[df_t["RA"].isin(shared)].set_index("RA")
        t1 = df_t1[df_t1["RA"].isin(shared)].set_index("RA")[["defasagem_raw"]]
        merged = t.join(t1, rsuffix="_next")
        # Target: did the student's defasagem get WORSE (more negative) next year?
        # y=1 → piorou (defasagem_next < defasagem_current)
        # y=0 → manteve ou melhorou
        y = (merged["defasagem_raw_next"] < merged["defasagem"]).astype("float32")

        # Drop rows with ANY null in feature source columns (systematic non-evaluation)
        available = [c for c in FEATURE_SOURCE_COLS if c in merged.columns]
        before = len(merged)
        mask_null = merged[available].isna().any(axis=1)
        dropped = int(mask_null.sum())
        merged = merged[~mask_null]
        y = y[~mask_null]
        if dropped:
            log.info(
                "%s: dropped %d/%d rows with null features (%.1f%%)", label, dropped, before, dropped / before * 100
            )
        return merged, y

    train_df, y_train = make_pairs(d22, d23, "Train(22→23)")
    test_df, y_test = make_pairs(d23, d24, "Test(23→24)")
    log.info(
        "Training pairs: %d (pos=%.1f%%)  |  Test pairs: %d (pos=%.1f%%)",
        len(train_df),
        y_train.mean() * 100,
        len(test_df),
        y_test.mean() * 100,
    )

    # ── 5. Inference set: impute ALL nulls (can't drop enrolled students) ─────
    # Imputation params fitted on train only to avoid leakage.
    inde_median = (
        float(train_df["INDE"].median()) if "INDE" in train_df.columns and not train_df["INDE"].isna().all() else 7.0
    )
    indicator_medians = {
        col: float(train_df[col].median())
        for col in ["IAA", "IEG", "IPS", "IDA", "IPV", "defasagem", "gender", "age"]
        if col in train_df.columns
    }
    # IPP: imputed from 2023 data for display only (not a model feature)
    _ipp_ref = d23.groupby("fase_num")["IPP"].median().to_dict() if "IPP" in d23.columns else {}
    _ipp_global = float(d23["IPP"].median()) if "IPP" in d23.columns and not d23["IPP"].isna().all() else 7.5

    d24_clean = d24.copy().reset_index(drop=True)
    # INDE
    if "INDE" not in d24_clean.columns:
        d24_clean["INDE"] = np.nan
    d24_clean["INDE"] = pd.to_numeric(d24_clean["INDE"], errors="coerce").fillna(inde_median)
    # Other model indicators
    for col, med in indicator_medians.items():
        if col in d24_clean.columns:
            d24_clean[col] = d24_clean[col].fillna(med)

    # Treat IEG=0 and IDA=0 as likely data-entry errors or missing records.
    # Strategy: save the original zero for frontend display, impute with phase
    # median from train set (fallback to global train median) for the model only.
    _ZERO_AS_MISSING = ["IEG", "IDA"]
    for _col in _ZERO_AS_MISSING:
        if _col in d24_clean.columns:
            d24_clean[f"{_col}_display"] = d24_clean[_col].copy()  # raw value for UI
            _zero_mask = d24_clean[_col] == 0.0
            if _zero_mask.any():
                _phase_med = train_df.groupby("fase_num")[_col].median().to_dict() if _col in train_df.columns else {}
                _global_med = float(train_df[_col].median()) if _col in train_df.columns else 7.0
                _imputed = d24_clean.loc[_zero_mask, "fase_num"].map(_phase_med).fillna(_global_med)
                d24_clean.loc[_zero_mask, _col] = _imputed
                log.info(
                    "Zero-imputation %s: %d students → phase/train medians (display preserves 0.0)",
                    _col,
                    int(_zero_mask.sum()),
                )

    # IPP for display only — impute with 2023 phase medians
    if "IPP" not in d24_clean.columns:
        d24_clean["IPP"] = np.nan
    ipp_mask = d24_clean["IPP"].isna()
    d24_clean.loc[ipp_mask, "IPP"] = d24_clean.loc[ipp_mask, "fase_num"].map(_ipp_ref).fillna(_ipp_global)

    # ── 5. Extract feature matrices ───────────────────────────────────────────
    def extract_X(df: pd.DataFrame) -> np.ndarray:
        return df[FEATURES].to_numpy(dtype="float32")

    X_train_raw = extract_X(train_df)
    X_test_raw = extract_X(test_df)
    X_infer_raw = extract_X(d24_clean)

    # ── 6. Scale — fit ONLY on train ──────────────────────────────────────────
    # RobustScaler: uses median + IQR — resistant to outliers in defasagem/INDE.
    # MinMaxScaler was sensitive to extreme values and could send inference
    # features outside [0, 1] when 2024 distribution differs from 2022.
    # We clip at ±5 IQR units: beyond that is a true outlier, not signal.
    _CLIP = 5.0
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = np.clip(scaler.transform(X_test_raw), -_CLIP, _CLIP)
    X_infer = np.clip(scaler.transform(X_infer_raw), -_CLIP, _CLIP)
    log.info(
        "Scaling done (RobustScaler) — train median=%.4f  IQR=%.4f  clip=±%.1f",
        np.median(X_train),
        np.subtract(*np.percentile(X_train, [75, 25])),
        _CLIP,
    )

    # ── 7. Student metadata for 2024 inference ────────────────────────────────
    students_meta = []
    for i, row in d24_clean.iterrows():
        nome = str(row.get("Nome", "")) or ""
        display_name = nome.split()[0] if nome.strip() and nome != "nan" else f"Aluno-{i}"
        fase_raw = row.get("Fase", "")
        fase_num = int(row.get("fase_num", 0))
        phase_label = str(fase_raw) if str(fase_raw) not in ("nan", "") else f"Fase {fase_num}"
        students_meta.append(
            {
                "student_id": int(i),
                "ra": str(row.get("RA", f"RA-{i}")),
                "display_name": display_name,
                "phase": phase_label,
                "fase_num": fase_num,
                "class_group": str(row.get("Turma", "N/A")),
                "gender": int(row.get("gender", 0)),
                "age": _safe_int(row.get("age")),
                "year": 2024,
                "iaa": _safe_float(row.get("IAA")),
                "ieg": _safe_float(row.get("IEG_display", row.get("IEG"))),  # original (may be 0)
                "ips": _safe_float(row.get("IPS")),
                "ida": _safe_float(row.get("IDA_display", row.get("IDA"))),  # original (may be 0)
                "ipv": _safe_float(row.get("IPV")),
                "ipp": _safe_float(row.get("IPP")),
                "inde": _safe_float(row.get("INDE")),
                "defasagem": _safe_int(row.get("defasagem")),
            }
        )

    # ── 8. Persist ────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _save(name: str, arr: np.ndarray) -> None:
        p = PROCESSED_DIR / name
        np.save(p, arr)
        log.info("Saved %s  shape=%s", name, arr.shape)

    _save("X_train.npy", X_train)
    _save("y_train.npy", y_train.to_numpy(dtype="float32"))
    _save("X_test.npy", X_test)
    _save("y_test.npy", y_test.to_numpy(dtype="float32"))
    _save("X_inference.npy", X_infer)

    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    log.info("Saved scaler.pkl")

    with open(PROCESSED_DIR / "students_meta.pkl", "wb") as f:
        pickle.dump(students_meta, f)
    log.info("Saved students_meta.pkl  (%d records)", len(students_meta))

    log.info(
        "ETL complete ✓  Train=%d  Test=%d  Inference=%d  INPUT_SIZE=%d",
        len(X_train),
        len(X_test),
        len(X_infer),
        INPUT_SIZE,
    )


if __name__ == "__main__":
    run_etl()
