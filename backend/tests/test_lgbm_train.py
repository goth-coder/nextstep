"""
Unit tests for the LightGBM training pipeline (train_lgbm.py).

Uses synthetic in-memory arrays to exercise the full pipeline without
touching real data files or a real MLflow server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_TRAIN = 120
N_TEST = 30
N_FEATURES = 10  # matches INPUT_SIZE after ETL


@pytest.fixture()
def synthetic_arrays(tmp_path):
    """Synthetic arrays written to a temp dir; patches PROCESSED_DIR."""
    rng = np.random.default_rng(42)

    y_train = rng.integers(0, 2, N_TRAIN).astype(float)
    y_test = rng.integers(0, 2, N_TEST).astype(float)
    # Ensure both classes present in train / test to avoid degenerate AUC
    y_train[: N_TRAIN // 5] = 0
    y_train[N_TRAIN // 5 : N_TRAIN // 5 * 2] = 1
    y_test[:5] = 0
    y_test[5:10] = 1

    X_train = rng.standard_normal((N_TRAIN, N_FEATURES)).astype(np.float32)
    X_test = rng.standard_normal((N_TEST, N_FEATURES)).astype(np.float32)

    np.save(tmp_path / "X_train.npy", X_train)
    np.save(tmp_path / "y_train.npy", y_train)
    np.save(tmp_path / "X_test.npy", X_test)
    np.save(tmp_path / "y_test.npy", y_test)

    return tmp_path, X_train, y_train, X_test, y_test


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_load_arrays_returns_six_splits(synthetic_arrays):
    """load_arrays() should return (X_tr, y_tr, X_val, y_val, X_test, y_test)."""
    tmp_path, X_train, y_train, X_test, y_test = synthetic_arrays

    import ml.train_lgbm as mod

    with patch.object(mod, "PROCESSED_DIR", tmp_path):
        splits = mod.load_arrays()

    assert len(splits) == 6
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits
    # val is last 20% of train
    expected_val_size = max(1, int(0.20 * N_TRAIN))
    assert len(y_val) == expected_val_size
    assert len(y_tr) == N_TRAIN - expected_val_size
    assert X_te.shape == (N_TEST, N_FEATURES)


def test_fit_calibrator_returns_platt(synthetic_arrays):
    """fit_calibrator() runs on a fitted LGBM and returns a PlattCalibrator."""
    import lightgbm as lgb

    import ml.train_lgbm as mod

    tmp_path, X_train, y_train, X_test, y_test = synthetic_arrays
    model = lgb.LGBMClassifier(n_estimators=10, verbosity=-1)
    model.fit(X_train, y_train.astype(int))

    X_val = X_train[-20:]
    y_val = y_train[-20:]

    cal = mod.fit_calibrator(model, X_val, y_val)

    # Check by class name to avoid module-path identity mismatch between
    # 'training.evaluator.PlattCalibrator' and 'ml.training.evaluator.PlattCalibrator'
    assert type(cal).__name__ == "PlattCalibrator"
    assert hasattr(cal, "predict_proba")
    probs = cal.predict_proba(np.zeros(5))
    assert probs.shape == (5,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_find_threshold_in_unit_range(synthetic_arrays):
    """find_threshold() should return threshold in [0, 1] and f1 in [0, 1]."""
    import lightgbm as lgb

    import ml.train_lgbm as mod

    tmp_path, X_train, y_train, _, _ = synthetic_arrays
    val_size = max(1, int(0.20 * len(X_train)))
    X_tr, X_val = X_train[: -val_size], X_train[-val_size:]
    y_tr, y_val = y_train[: -val_size], y_train[-val_size:]

    model = lgb.LGBMClassifier(n_estimators=10, verbosity=-1)
    model.fit(X_tr, y_tr.astype(int))
    cal = mod.fit_calibrator(model, X_val, y_val)

    threshold, val_f1 = mod.find_threshold(model, X_val, y_val, cal)

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= val_f1 <= 1.0


def test_train_lgbm_quality_gate_skip(synthetic_arrays, monkeypatch):
    """
    train_lgbm() should complete without sys.exit when the quality gate is bypassed.
    MLflow is fully mocked so no server is needed.
    """
    tmp_path, _, _, _, _ = synthetic_arrays

    import ml.train_lgbm as mod

    # Bypass quality gate so the test does not fail on synthetic data
    monkeypatch.setattr(mod, "MIN_AUC", 0.0)
    monkeypatch.setattr(mod, "MIN_F1", 0.0)
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)
    # Prevent scaler artifact logging (file doesn't exist)
    monkeypatch.setattr(mod, "SCALER_PATH", tmp_path / "nonexistent_scaler.pkl")

    # Proper context manager mock: __enter__ returns object with .info.run_id
    mock_run_ctx = MagicMock()
    mock_run_ctx.__enter__.return_value.info.run_id = "fake-run-id"
    fake_mv = MagicMock()
    fake_mv.version = "1"

    with (
        patch("ml.train_lgbm.mlflow.set_tracking_uri"),
        patch("ml.train_lgbm.mlflow.set_experiment"),
        patch("ml.train_lgbm.mlflow.start_run", return_value=mock_run_ctx),
        patch("ml.train_lgbm.mlflow.set_tag"),
        patch("ml.train_lgbm.mlflow.log_params"),
        patch("ml.train_lgbm.mlflow.log_metrics"),
        patch("ml.train_lgbm.mlflow.log_artifact"),
        patch("ml.train_lgbm.mlflow.lightgbm.log_model"),
        patch("ml.train_lgbm.mlflow.register_model", return_value=fake_mv),
        patch("ml.train_lgbm.MlflowClient") as mock_client_cls,
    ):
        mock_client_cls.return_value = MagicMock()
        result = mod.train_lgbm(params={"n_estimators": 10})

    # EvalResult fields should be finite floats
    assert 0.0 <= result.val_auc <= 1.0
    assert 0.0 <= result.val_f1 <= 1.0
    assert 0.0 <= result.threshold <= 1.0


def test_train_lgbm_quality_gate_exits(synthetic_arrays, monkeypatch):
    """
    train_lgbm() should call sys.exit(1) when quality gate thresholds are impossibly high.
    MLflow is fully mocked — the gate runs BEFORE logging, so no MLflow calls happen.
    """
    tmp_path, _, _, _, _ = synthetic_arrays

    import ml.train_lgbm as mod

    # Set impossible thresholds so quality gate always fails
    monkeypatch.setattr(mod, "MIN_AUC", 1.1)
    monkeypatch.setattr(mod, "MIN_F1", 1.1)
    monkeypatch.setattr(mod, "PROCESSED_DIR", tmp_path)

    with (
        patch("ml.train_lgbm.mlflow.set_tracking_uri"),
        patch("ml.train_lgbm.mlflow.set_experiment"),
        patch("ml.train_lgbm.mlflow.start_run"),
        patch("ml.train_lgbm.MlflowClient"),
        pytest.raises(SystemExit) as exc_info,
    ):
        mod.train_lgbm(params={"n_estimators": 10})

    assert exc_info.value.code == 1
