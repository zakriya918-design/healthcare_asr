"""Automated build script for packaging Healthcare ASR into a standalone Windows executable.

Uses PyInstaller to bundle Python runtime, Tkinter GUI, Faster-Whisper, and assets.
Usage:
    python scripts/build_exe.py [--onedir | --onefile]
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildExe")


def build_executable(onefile: bool = False):
    root = get_project_root()
    app_py = root / "app.py"
    dist_dir = root / "dist"
    build_dir = root / "build"

    logger.info("Checking PyInstaller availability...")
    try:
        import PyInstaller
    except ImportError:
        logger.info("Installing pyinstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name", "HealthcareASR",
        "--noconsole",
        "--clean",
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
        # Bundle data
        "--add-data", f"{root / 'src'};src",
        # Hidden imports needed for dynamic reflection
        "--hidden-import", "sounddevice",
        "--hidden-import", "soundfile",
        "--hidden-import", "reportlab",
        "--hidden-import", "reportlab.lib.pagesizes",
        "--hidden-import", "reportlab.platypus",
        "--hidden-import", "ctranslate2",
        "--hidden-import", "faster_whisper",
        "--hidden-import", "src.utils.paths",
        "--hidden-import", "src.processing.archive_exporter",
        "--hidden-import", "src.processing.fhir_exporter",
        "--hidden-import", "src.processing.icd10_coding",
        str(app_py)
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    logger.info(f"Running PyInstaller command:\n{' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode == 0:
        logger.info(f"Build succeeded! Executable generated in: {dist_dir / 'HealthcareASR'}")
    else:
        logger.error(f"Build failed with exit code: {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Healthcare ASR Assistant into a Windows .exe")
    parser.add_argument("--onefile", action="store_true", help="Package into a single .exe file instead of folder")
    args = parser.parse_args()

    build_executable(onefile=args.onefile)
