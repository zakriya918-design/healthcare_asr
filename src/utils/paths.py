"""Path utilities for Healthcare AI Clinical Documentation Assistant.

Provides dynamic, environment-aware, cross-platform path resolution.
Eliminates hardcoded user directories to ensure seamless deployment on any machine.
"""

from __future__ import annotations
import os
from pathlib import Path


def get_project_root() -> Path:
    """Returns the root directory of the application repository."""
    # src/utils/paths.py -> parent is utils, parent is src, parent is root
    return Path(__file__).resolve().parent.parent.parent


def get_base_data_dir() -> Path:
    """
    Returns the primary base directory for storing application data.
    Order of precedence:
    1. Environment variable HEALTHCARE_ASR_DATA_DIR
    2. Path.home() / "HealthcareASR"
    """
    env_dir = os.environ.get("HEALTHCARE_ASR_DATA_DIR")
    if env_dir:
        base = Path(env_dir)
    else:
        base = Path.home() / "HealthcareASR"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_output_dir() -> Path:
    """Returns the directory for generated consultation records, HTML, PDF, and ZIP files."""
    out = get_base_data_dir() / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_recordings_dir() -> Path:
    """Returns the directory for saved audio WAV consultation recordings."""
    rec = get_base_data_dir() / "recordings"
    rec.mkdir(parents=True, exist_ok=True)
    return rec


def get_models_dir() -> Path:
    """Returns the directory for offline local Faster-Whisper model weights."""
    # Check project-relative models/ first, then base data dir
    local_proj_models = get_project_root() / "models"
    if local_proj_models.exists():
        return local_proj_models
    models = get_base_data_dir() / "models"
    models.mkdir(parents=True, exist_ok=True)
    return models


def get_synthetic_wav_path() -> Path:
    """Returns the path to the synthetic consultation audio WAV file."""
    # 1. Check inside project fixtures
    proj_fixture = get_project_root() / "recordings" / "synthetic_consultation_test.wav"
    if proj_fixture.is_file():
        return proj_fixture
    # 2. Check base recordings directory
    base_fixture = get_recordings_dir() / "synthetic_consultation_test.wav"
    return base_fixture
