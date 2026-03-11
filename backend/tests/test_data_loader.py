"""
Unit tests for ETL pipeline (data_loader.py).

Tests mock _load_sheet() to inject synthetic DataFrames with the correct
schema, avoiding a dependency on a real XLSX/GCS file.

The synthetic sheets cover years 2022, 2023, 2024 with shared RA keys
so the temporal pair-building logic can be exercised end-to-end.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml.data_loader import INPUT_SIZE  # noqa: E402

# ── Synthetic data factory ────────────────────────────────────────────────────

def _make_sheet(
    n: int, year: int, ras: list[str], rng: np.random.Generator, null_fraction: float = 0.1
) -> pd.DataFrame:
    """Return a synthetic DataFrame matching the schema produced by _load_sheet()."""
    data = {
        "RA": ras,
        "Nome": [f"ALUNO-{ra}" for ra in ras],
        "Fase": rng.integers(1, 6, n).astype(str),
        "Turma": rng.choice(["A", "B", "C", "G", "F"], n),
        "IAA": rng.uniform(4, 10, n),
        "IEG": rng.uniform(3, 10, n),
        "IPS": rng.uniform(4, 10, n),
        "IDA": rng.uniform(3, 10, n),
        "IPV": rng.uniform(4, 10, n),
        "IPP": rng.uniform(4, 10, n),
        "INDE": rng.uniform(4, 10, n),
        "defasagem": rng.integers(-3, 2, n).astype(float),
        "defasagem_raw": rng.integers(-3, 2, n).astype(float),
        "fase_num": rng.integers(0, 7, n),
        "gender": rng.integers(0, 2, n),
        "age": rng.uniform(8, 18, n),
        "mat": rng.uniform(3, 10, n),
        "por": rng.uniform(3, 10, n),
        "tenure": rng.integers(1, 6, n).astype(float),
        "n_av": rng.integers(0, 4, n).astype(float),
        "IAN": rng.uniform(4, 10, n),
        "year": year,
    }
    df = pd.DataFrame(data)

    # Introduce a few NaN values in indicator columns (real data has gaps)
    for col in ["IAA", "IEG", "IPS", "IDA", "IPV"]:
        null_mask = rng.random(n) < null_fraction
        df.loc[null_mask, col] = np.nan

    return df


def _make_sheets(n: int = 30, null_fraction: float = 0.1):
    """Return three synthetic sheets (2022, 2023, 2024) with overlapping RAs."""
    rng = np.random.default_rng(42)
    # Use same RAs across years so temporal pairs can be built
    ras = [f"RA-{i:04d}" for i in range(1, n + 1)]
    d22 = _make_sheet(n, 2022, ras, rng, null_fraction)
    d23 = _make_sheet(n, 2023, ras, rng, null_fraction)
    d24 = _make_sheet(n, 2024, ras, rng, null_fraction=0)  # no nulls for inference
    return d22, d23, d24


# ── ETL runner with patches ───────────────────────────────────────────────────

def _run_etl(processed_dir: Path, n: int = 30, null_fraction: float = 0.1) -> tuple:
    """Run the full ETL pipeline with synthetic sheets injected via mocks."""
    d22, d23, d24 = _make_sheets(n=n, null_fraction=null_fraction)

    # _load_sheet returns different frames per year argument
    def fake_load_sheet(xl, sheet_name: str, year: int) -> pd.DataFrame:
        mapping = {2022: d22, 2023: d23, 2024: d24}
        return mapping[year]

    fake_xl = MagicMock()  # ExcelFile — not actually read

    with (
        patch("ml.data_loader._find_xlsx", return_value="fake.xlsx"),
        patch("ml.data_loader.pd.ExcelFile", return_value=fake_xl),
        patch("ml.data_loader._load_sheet", side_effect=fake_load_sheet),
        patch("ml.data_loader.PROCESSED_DIR", processed_dir),
        patch("ml.data_loader.os.environ.get", side_effect=lambda k, d="": "" if k == "GCS_BUCKET" else d),
    ):
        from ml.data_loader import run_etl
        run_etl()

    return d22, d23, d24


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def etl_result(tmp_path: Path):
    """Run ETL once, return processed_dir + original sheets."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    d22, d23, d24 = _run_etl(processed_dir)
    return {"processed_dir": processed_dir, "d22": d22, "d23": d23, "d24": d24}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_output_files_created(etl_result):
    """All expected output files must exist after ETL."""
    pd = etl_result["processed_dir"]
    for fname in ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy",
                  "X_inference.npy", "scaler.pkl", "students_meta.pkl"]:
        assert (pd / fname).exists(), f"{fname} was not created by ETL"


def test_train_tensor_shape(etl_result):
    """X_train must be (N, INPUT_SIZE); y_train must be (N,)."""
    pd_dir = etl_result["processed_dir"]
    X = np.load(pd_dir / "X_train.npy")
    y = np.load(pd_dir / "y_train.npy")
    assert X.ndim == 2, f"Expected 2D, got {X.shape}"
    assert X.shape[1] == INPUT_SIZE, f"Expected INPUT_SIZE={INPUT_SIZE}, got {X.shape[1]}"
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0], "X_train and y_train length mismatch"


def test_test_tensor_shape(etl_result):
    """X_test and y_test must have the same number of rows and INPUT_SIZE features."""
    pd_dir = etl_result["processed_dir"]
    X = np.load(pd_dir / "X_test.npy")
    y = np.load(pd_dir / "y_test.npy")
    assert X.shape[1] == INPUT_SIZE
    assert X.shape[0] == y.shape[0]


def test_inference_tensor_shape(etl_result):
    """X_inference must cover all 2024 students with INPUT_SIZE features."""
    pd_dir = etl_result["processed_dir"]
    X = np.load(pd_dir / "X_inference.npy")
    assert X.ndim == 2
    assert X.shape[1] == INPUT_SIZE
    # 2024 sheet has 30 students (no drops — null_fraction=0)
    assert X.shape[0] == 30, f"Expected 30 inference rows, got {X.shape[0]}"


def test_no_nan_in_train(etl_result):
    """X_train must have no NaN — null rows are dropped during pair-building."""
    X = np.load(etl_result["processed_dir"] / "X_train.npy")
    assert not np.isnan(X).any(), "X_train contains NaN values"


def test_no_nan_in_inference(etl_result):
    """X_inference must have no NaN — inference uses median imputation."""
    X = np.load(etl_result["processed_dir"] / "X_inference.npy")
    assert not np.isnan(X).any(), "X_inference contains NaN values (imputation failed)"


def test_train_scaled_median_near_zero(etl_result):
    """After RobustScaler fit on train, the median of X_train should be ≈0."""
    X = np.load(etl_result["processed_dir"] / "X_train.npy")
    assert abs(np.median(X)) < 1.0, f"Train median far from 0: {np.median(X):.4f}"


def test_scaler_persists_and_reloads(etl_result):
    """scaler.pkl must be a valid fitted RobustScaler with INPUT_SIZE features."""
    scaler_path = etl_result["processed_dir"] / "scaler.pkl"
    assert scaler_path.exists()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    from sklearn.preprocessing import RobustScaler
    assert isinstance(scaler, RobustScaler)
    assert scaler.n_features_in_ == INPUT_SIZE, (
        f"Expected scaler fitted on {INPUT_SIZE} features, got {scaler.n_features_in_}"
    )


def test_students_meta_has_correct_count(etl_result):
    """students_meta.pkl should have one record per 2024 student."""
    meta_path = etl_result["processed_dir"] / "students_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    assert len(meta) == 30, f"Expected 30 student records, got {len(meta)}"


def test_students_meta_fields(etl_result):
    """Each record must have the mandatory fields consumed by the API."""
    meta_path = etl_result["processed_dir"] / "students_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    required_fields = {"student_id", "display_name", "phase", "class_group", "iaa", "ieg", "ips", "ida", "ipv"}
    for record in meta:
        missing = required_fields - record.keys()
        assert not missing, f"Record {record.get('student_id')} missing fields: {missing}"


def test_binary_labels(etl_result):
    """All y values must be 0.0 or 1.0 (binary classification)."""
    for fname in ["y_train.npy", "y_test.npy"]:
        y = np.load(etl_result["processed_dir"] / fname)
        unique = set(np.unique(y))
        assert unique <= {0.0, 1.0}, f"{fname} has unexpected label values: {unique}"
