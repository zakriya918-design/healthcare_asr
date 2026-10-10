"""Speech-to-Text transcription module using Faster-Whisper.

DISCLAIMER:
Uses Faster-Whisper CPU inference. Models are loaded lazily on demand.
Never fabricates transcript text when recognition fails.
"""

from __future__ import annotations
import logging
from pathlib import Path
import time
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
from src.models.patient_record import TranscriptSegment

logger = logging.getLogger(__name__)


class ASRError(Exception):
    """Base exception for ASR transcription errors."""
    pass


class ModelLoadError(ASRError):
    """Raised when ASR model fails to download or initialize."""
    pass


class TranscriptionError(ASRError):
    """Raised when transcription fails during inference."""
    pass


def load_and_resample_audio(audio_path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
    """
    Safely loads audio using soundfile and converts to 16 kHz mono float32 numpy array.
    Avoids native container/PyAV version bugs by feeding decoded waveforms directly to Whisper.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise TranscriptionError(f"Audio file not found: {path}")

    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception as e:
        raise TranscriptionError(f"Cannot read audio file {path.name}: {e}") from e

    if data.size == 0 or len(data) < int(sr * 0.1):
        raise TranscriptionError(f"Audio file {path.name} is empty or too short.")

    # Convert multi-channel / stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16,000 Hz if necessary using linear interpolation
    if sr != target_sr:
        target_len = int(len(data) * target_sr / sr)
        if target_len > 0:
            data = np.interp(
                np.linspace(0, len(data), target_len),
                np.arange(len(data)),
                data
            ).astype(np.float32)

    return data


class WhisperTranscriber:
    """
    Wrapper for Faster-Whisper with CPU-compatible inference and lazy loading.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[Path] = None
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._model = None
        self._is_loading = False

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Lazily loads the Whisper model.
        Safe for use inside background worker threads.
        """
        if self._model is not None:
            return

        if progress_callback:
            progress_callback(f"Loading Faster-Whisper model ({self.model_size}) on CPU...")

        try:
            from faster_whisper import WhisperModel
            from src.utils.paths import get_models_dir

            # Check for pre-bundled offline model folder
            models_dir = get_models_dir()
            local_candidate = models_dir / self.model_size
            if local_candidate.is_dir() and (local_candidate / "model.bin").is_file():
                model_target = str(local_candidate)
                logger.info(f"Loading Faster-Whisper from pre-bundled offline directory: {model_target}")
            else:
                model_target = self.model_size

            root_dir = str(self.download_root) if self.download_root else str(models_dir)
            self._model = WhisperModel(
                model_size_or_path=model_target,
                device=self.device,
                compute_type=self.compute_type,
                download_root=root_dir
            )
            logger.info(f"Faster-Whisper model '{self.model_size}' successfully loaded on {self.device}.")
            if progress_callback:
                progress_callback(f"Model {self.model_size} ready.")
        except Exception as e:
            self._model = None
            raise ModelLoadError(
                f"Failed to load Faster-Whisper model '{self.model_size}': {e}. "
                "Verify internet connectivity for initial download or check disk space."
            ) from e

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[str, List[TranscriptSegment]]:
        """
        Transcribes an audio file (WAV, FLAC, etc.) into full text and structured segments.
        Returns:
            (full_transcript_text, list_of_TranscriptSegment)
        """
        # Load and validate waveform via soundfile
        audio_waveform = load_and_resample_audio(audio_path, target_sr=16000)

        # Ensure model is loaded
        if self._model is None:
            self.load_model(progress_callback)

        path = Path(audio_path)
        if progress_callback:
            progress_callback(f"Transcribing {path.name}...")

        try:
            # Transcribe the float32 waveform directly
            segments_gen, info = self._model.transcribe(
                audio_waveform,
                beam_size=5,
                language="en",
                condition_on_previous_text=False
            )

            full_text_pieces = []
            segments: List[TranscriptSegment] = []
            seg_id = 1

            for segment in segments_gen:
                text = segment.text.strip()
                if text:
                    full_text_pieces.append(text)
                    segments.append(TranscriptSegment(
                        id=seg_id,
                        text=text,
                        start_time=round(segment.start, 2),
                        end_time=round(segment.end, 2)
                    ))
                    seg_id += 1

            full_transcript = " ".join(full_text_pieces).strip()

            if progress_callback:
                progress_callback("Transcription complete.")

            return full_transcript, segments

        except Exception as e:
            raise TranscriptionError(f"Transcription failed for {path.name}: {e}") from e
