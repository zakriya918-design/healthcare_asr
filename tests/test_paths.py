"""Unit tests for dynamic paths utility."""

import os
from pathlib import Path
from src.utils.paths import (
    get_project_root,
    get_base_data_dir,
    get_output_dir,
    get_recordings_dir,
    get_models_dir,
    get_synthetic_wav_path
)


def test_project_root():
    root = get_project_root()
    assert root.is_dir()
    assert (root / "app.py").is_file()


def test_base_data_dir_default():
    base = get_base_data_dir()
    assert base.is_dir()
    assert "HealthcareASR" in str(base)


def test_base_data_dir_env_override(monkeypatch, tmp_path):
    custom_dir = tmp_path / "custom_clinic_data"
    monkeypatch.setenv("HEALTHCARE_ASR_DATA_DIR", str(custom_dir))
    base = get_base_data_dir()
    assert base == custom_dir
    assert custom_dir.is_dir()


def test_subdirectories(monkeypatch, tmp_path):
    monkeypatch.setenv("HEALTHCARE_ASR_DATA_DIR", str(tmp_path))
    out = get_output_dir()
    rec = get_recordings_dir()
    models = get_models_dir()

    assert out == tmp_path / "output"
    assert rec == tmp_path / "recordings"
    assert out.is_dir()
    assert rec.is_dir()
    assert models.is_dir()


def test_synthetic_wav_path():
    path = get_synthetic_wav_path()
    assert isinstance(path, Path)
    assert path.name == "synthetic_consultation_test.wav"
