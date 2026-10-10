"""Pre-download Faster-Whisper models for offline air-gapped clinical deployment."""

import argparse
import logging
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelDownloader")


def download_model(model_name: str = "tiny.en", dest_dir: Path = None):
    try:
        from faster_whisper import download_model as fw_download
    except ImportError:
        logger.error("faster_whisper is not installed. Run: pip install faster-whisper")
        sys.exit(1)

    if dest_dir is None:
        dest_dir = get_project_root() / "models" / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Faster-Whisper model '{model_name}' into {dest_dir}...")
    saved_path = fw_download(model_name, output_dir=str(dest_dir))
    logger.info(f"Model successfully downloaded to: {saved_path}")
    return Path(saved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Faster-Whisper models for offline bundling.")
    parser.add_argument("--model", default="tiny.en", choices=["tiny.en", "base.en", "small.en"], help="Model name")
    parser.add_argument("--dest", type=Path, default=None, help="Destination directory")
    args = parser.parse_args()

    download_model(model_name=args.model, dest_dir=args.dest)
