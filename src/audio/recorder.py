"""Audio recording module using sounddevice for Healthcare AI Clinical Documentation Assistant.

WORKFLOW DOCUMENTATION:
- Recording: Captures raw mono 16 kHz audio chunks via sounddevice into a thread-safe memory queue.
- Transcription after recording: Upon calling stop_recording(), audio is assembled, validated, saved
  to a local WAV file in the recordings/ folder, and passed to Faster-Whisper for batch transcription.
- This is NOT streaming/live transcription; it is recording followed by post-recording ASR.
"""

from __future__ import annotations
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioRecorderError(Exception):
    """Base exception for audio recorder errors."""
    pass


class AudioDeviceError(AudioRecorderError):
    """Raised when no input device is available or permission is denied."""
    pass


class AudioValidationError(AudioRecorderError):
    """Raised when recorded audio is invalid, empty, or corrupted."""
    pass


class AudioRecorder:
    """
    Manages mono audio recording at 16,000 Hz.
    Provides non-blocking thread-safe start, stop, and cancel controls.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[Union[int, str]] = None,
        recordings_dir: Optional[Path] = None,
        max_duration_seconds: int = 3600  # Safety limit: 1 hour maximum
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.max_duration_seconds = max_duration_seconds
        self.recordings_dir = recordings_dir or Path(r"C:\Users\Aziza\HEALTHCARE_ASR\recordings")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._is_recording = False
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    @staticmethod
    def get_available_input_devices() -> list:
        """Returns list of available audio input devices with device index and name."""
        try:
            devices = sd.query_devices()
            input_devices = []
            for idx, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    input_devices.append({
                        "index": idx,
                        "name": str(d.get("name", f"Device {idx}")),
                        "channels": int(d.get("max_input_channels", 1)),
                        "default_samplerate": float(d.get("default_samplerate", 16000))
                    })
            return input_devices
        except Exception as e:
            logger.warning(f"Failed to query sound devices: {e}")
            return []

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback executed by PortAudio in a separate audio thread."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        if self._is_recording:
            # Copy data to prevent buffer overwrite
            self._audio_queue.put(indata.copy())

    def start_recording(self) -> None:
        """Starts recording audio from designated or default input device."""
        with self._lock:
            if self._is_recording:
                raise AudioRecorderError("Recording is already in progress.")

            # Validate input devices
            devices = self.get_available_input_devices()
            if not devices:
                raise AudioDeviceError(
                    "No audio input device (microphone) detected on this system. "
                    "Please connect a microphone or use WAV file import."
                )

            # Clear queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    device=self.device,
                    dtype="float32",
                    callback=self._audio_callback
                )
                self._stream.start()
                self._is_recording = True
                self._start_time = time.time()
                dev_label = f"device index {self.device}" if self.device is not None else "default device"
                logger.info(f"Audio recording started at {self.sample_rate} Hz mono on {dev_label}.")
            except Exception as e:
                self._is_recording = False
                if self._stream:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                raise AudioDeviceError(f"Failed to start audio recording stream: {e}") from e

    def stop_recording(self, destination_path: Optional[Path] = None) -> Path:
        """
        Stops recording, collects all audio chunks, validates duration and audio content,
        and saves to a 16 kHz WAV file.
        """
        with self._lock:
            if not self._is_recording:
                raise AudioRecorderError("Cannot stop recording: No active recording.")

            self._is_recording = False
            duration_est = time.time() - (self._start_time or time.time())

            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"Error while closing audio stream: {e}")
                finally:
                    self._stream = None

        # Drain audio chunks from queue
        chunks = []
        while not self._audio_queue.empty():
            try:
                chunks.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break

        if not chunks:
            raise AudioValidationError("Recorded audio is empty (no audio frames captured).")

        audio_data = np.concatenate(chunks, axis=0)

        # Validate non-empty content
        if audio_data.size == 0 or len(audio_data) < int(self.sample_rate * 0.2):
            raise AudioValidationError(
                f"Audio recording too short ({len(audio_data) / self.sample_rate:.2f}s). "
                "Minimum duration is 0.2 seconds."
            )

        # Generate unique destination path if none specified
        if destination_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dest = self.recordings_dir / f"consultation_rec_{timestamp}.wav"
        else:
            dest = Path(destination_path)

        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            sf.write(str(dest), audio_data, self.sample_rate, subtype="PCM_16")
            logger.info(f"Recorded audio saved: {dest} (frames: {len(audio_data)}, duration: {len(audio_data)/self.sample_rate:.2f}s)")
            return dest
        except Exception as e:
            raise AudioRecorderError(f"Failed to save audio recording to {dest}: {e}") from e

    def cancel_recording(self) -> None:
        """Stops recording and cleanly discards any captured audio."""
        with self._lock:
            self._is_recording = False
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                finally:
                    self._stream = None

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Audio recording cancelled and buffers cleared.")


def validate_wav_file(wav_path: Path) -> Tuple[bool, str, float]:
    """
    Validates that a WAV file exists, is valid audio, and has non-zero duration.
    Returns: (is_valid, message, duration_seconds)
    """
    if not wav_path.is_file():
        return False, f"Audio file does not exist: {wav_path}", 0.0

    try:
        info = sf.info(str(wav_path))
        if info.frames == 0 or info.duration < 0.1:
            return False, "Audio file is empty or shorter than 0.1s.", 0.0
        return True, "Valid audio file", info.duration
    except Exception as e:
        return False, f"Invalid or corrupted audio file: {e}", 0.0
