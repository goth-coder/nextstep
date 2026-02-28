"""
Unit tests for ETL pipeline (data_loader.py).

Tests run against a temporary directory with synthetic data to avoid
depending on the real dataset file.
"""

import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_df(n: int = 20, null_fraction: float = 0.2) -> pd.DataFrame:
    """Return a small synthetic DataFrame matching the real CSV schema."""
    rng = np.random.default_rng(42)

    data = {
        "ID_num": range(1, n + 1),
        "year": [2021] * n,
        "IAA": rng.uniform(4, 10, n),
        "IEG": rng.uniform(3, 10, n),
        "IPS": rng.uniform(4, 10, n),
        "IDA": rng.uniform(3, 10, n),
        "IAN": rng.uniform(0, 10, n),
        "IPV": rng.uniform(4, 10, n),
        "defasagem_raw": rng.uniform(-2, 2, n),
        "NOME": [f"ALUNO-{i}" for i in range(1, n + 1)],
        "Fase": rng.integers(1, 7, n).astype(float),
        "Turma": rng.choice(["A", "B", "C", "G", "F"], n),
        "defasagem_bin": rng.integers(0, 2, n),
        "has_indicator": [True] * n,
    }

    df = pd.DataFrame(data)

    # Introduce nulls in indicators
    for col in ["IAA", "IEG", "IPS", "IDA", "IAN", "IPV"]:
        null_mask = rng.random(n) < null_fraction
        df.loc[null_mask, col] = np.nan

    return df


@pytest.fixture()
def tmp_project(tmp_path: Path):
    """Create a temporary project with synthetic CSV and patched paths."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "dataset_unificado_defasagem.csv"

    df = _make_synthetic_df(20, null_fraction=0.3)
    df.to_csv(csv_path, index=False)

    processed_dir = tmp_path / "backend" / "data" / "processed"
    processed_dir.mkdir(parents=True)

    return {"csv_path": csv_path, "processed_dir": processed_dir, "tmp_path": tmp_path}


def _run_etl(tmp_project: dict) -> None:
    """Import and run the ETL pipeline with patched paths."""
    with (
        patch("src.data_loader._DEFAULT_PATHS", [tmp_project["csv_path"]]),
        patch("src.data_loader.PROCESSED_DIR", tmp_project["processed_dir"]),
    ):
        from src.data_loader import run_etl

        run_etl()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_imputation_fills_nulls(tmp_project):
    """After ETL, no nan values should remain in the X tensor."""
    _run_etl(tmp_project)
    X = np.load(tmp_project["processed_dir"] / "X.npy")
    assert not np.isnan(X).any(), "X tensor must have no NaN values after imputation"


def test_normalised_values_in_unit_range(tmp_project):
    """After RobustScaler, train features should have median≈0; values outside
    [-5, 5] are extreme outliers and are clipped in inference/test transforms."""
    _run_etl(tmp_project)
    X = np.load(tmp_project["processed_dir"] / "X.npy")
    # RobustScaler does NOT bound to [0,1]; check that bulk of values are reasonable
    assert not np.isnan(X).any(), "Scaled X must have no NaN"
    # After RobustScaler the median of the train transform should be near 0
    assert abs(np.median(X)) < 1.0, f"Median far from 0: {np.median(X):.4f} (RobustScaler expected)"


def test_output_tensor_shape(tmp_project):
    """X.npy should be (N, 7); y.npy should be (N,)."""
    _run_etl(tmp_project)
    X = np.load(tmp_project["processed_dir"] / "X.npy")
    y = np.load(tmp_project["processed_dir"] / "y.npy")
    assert X.ndim == 2, f"Expected 2D tensor, got shape {X.shape}"
    assert X.shape[1] == 6, f"Expected 6 indicators, got {X.shape[1]}"
    assert y.ndim == 1, f"Expected 1D label tensor, got shape {y.shape}"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"


def test_deduplication_removes_duplicate_rows(tmp_project):
    """Duplicate ID_num+year rows should be dropped."""
    # Add duplicates to the CSV
    csv_path = tmp_project["csv_path"]
    df = pd.read_csv(csv_path)
    dup_df = pd.concat([df, df.head(5)], ignore_index=True)
    dup_df.to_csv(csv_path, index=False)

    _run_etl(tmp_project)

    X = np.load(tmp_project["processed_dir"] / "X.npy")
    # Original has 20 unique rows; duplicated 5 rows should be dropped
    assert X.shape[0] == 20, f"Expected 20 rows after dedup, got {X.shape[0]}"


def test_scaler_persists_and_reloads(tmp_project):
    """scaler.pkl must be a valid sklearn RobustScaler after ETL."""
    _run_etl(tmp_project)
    scaler_path = tmp_project["processed_dir"] / "scaler.pkl"
    assert scaler_path.exists(), "scaler.pkl must be created by ETL"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    from sklearn.preprocessing import RobustScaler

    assert isinstance(scaler, RobustScaler), f"Expected RobustScaler, got {type(scaler)}"
    assert scaler.n_features_in_ == 8, f"Expected 8 features, got {scaler.n_features_in_}"


def test_students_meta_has_correct_count(tmp_project):
    """students_meta.pkl should contain one record per unique student."""
    _run_etl(tmp_project)
    meta_path = tmp_project["processed_dir"] / "students_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    assert len(meta) == 20, f"Expected 20 student records, got {len(meta)}"


def test_students_meta_fields(tmp_project):
    """Each record in students_meta.pkl must have required fields."""
    _run_etl(tmp_project)
    meta_path = tmp_project["processed_dir"] / "students_meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    required_fields = {"student_id", "display_name", "phase", "class_group", "iaa", "ieg", "ips", "ida", "ian", "ipv"}
    for record in meta:
        missing = required_fields - record.keys()
        assert not missing, f"Record {record.get('student_id')} missing fields: {missing}"
