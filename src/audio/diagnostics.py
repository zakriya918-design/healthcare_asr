"""Audio Pre-Flight Diagnostic Tool for Medical Consultations.

Analyzes microphone input quality:
- Root Mean Square (RMS) dBFS level
- Peak amplitude dBFS
- Estimated noise floor and Signal-to-Noise Ratio (SNR)
- Digital clipping detection
- Silence detection
- Clinician-facing quality rating and actionable hardware recommendations
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class AudioDiagnosticResult:
    """Detailed audio diagnostic report for pre-consultation hardware checks."""
    device_name: str
    device_index: Optional[int]
    sample_rate: int
    duration_seconds: float
    rms_db: float
    peak_db: float
    noise_floor_db: float
    snr_db: float
    clipping_ratio: float
    is_silent: bool
    has_clipping: bool
    quality_rating: str
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_name": self.device_name,
            "device_index": self.device_index,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 2),
            "rms_db": round(self.rms_db, 1),
            "peak_db": round(self.peak_db, 1),
            "noise_floor_db": round(self.noise_floor_db, 1),
            "snr_db": round(self.snr_db, 1),
            "clipping_ratio": round(self.clipping_ratio, 4),
            "is_silent": self.is_silent,
            "has_clipping": self.has_clipping,
            "quality_rating": self.quality_rating,
            "recommendations": list(self.recommendations),
        }


def _amplitude_to_db(amp: float, floor_db: float = -96.0) -> float:
    """Convert normalized linear amplitude (0.0 to 1.0) to dBFS."""
    if amp <= 1e-5:
        return floor_db
    return max(floor_db, 20.0 * math.log10(amp))


def analyze_audio_data(
    audio: np.ndarray,
    sample_rate: int = 16000,
    device_name: str = "Default Microphone",
    device_index: Optional[int] = None
) -> AudioDiagnosticResult:
    """Perform deterministic signal analysis on an audio sample array.
    
    Args:
        audio: 1D numpy array of audio samples (float32 normalized to [-1.0, 1.0]).
        sample_rate: Sampling frequency in Hz.
        device_name: Hardware name of audio input.
        device_index: Optional PortAudio device index.
        
    Returns:
        AudioDiagnosticResult containing acoustic metrics and clinical recommendations.
    """
    if audio.ndim > 1:
        audio = audio.flatten()

    total_samples = len(audio)
    duration_sec = total_samples / float(sample_rate) if sample_rate > 0 else 0.0

    if total_samples == 0:
        return AudioDiagnosticResult(
            device_name=device_name,
            device_index=device_index,
            sample_rate=sample_rate,
            duration_seconds=0.0,
            rms_db=-96.0,
            peak_db=-96.0,
            noise_floor_db=-96.0,
            snr_db=0.0,
            clipping_ratio=0.0,
            is_silent=True,
            has_clipping=False,
            quality_rating="CRITICAL",
            recommendations=["No audio data captured. Verify microphone permissions and connection."]
        )

    # 1. Peak & RMS Amplitude
    peak_linear = float(np.max(np.abs(audio)))
    rms_linear = float(np.sqrt(np.mean(np.square(audio))))

    peak_db = _amplitude_to_db(peak_linear)
    rms_db = _amplitude_to_db(rms_linear)

    # 2. Digital Clipping (samples at or exceeding 0.985 linear)
    clip_thresh = 0.985
    clipped_samples = int(np.sum(np.abs(audio) >= clip_thresh))
    clipping_ratio = float(clipped_samples / total_samples)
    has_clipping = clipping_ratio > 0.001  # > 0.1% clipped

    # 3. Noise Floor Estimation via windowed percentiles (100ms frames)
    frame_size = max(160, int(sample_rate * 0.1))  # 100ms
    num_frames = total_samples // frame_size

    if num_frames >= 3:
        frame_rms = [
            float(np.sqrt(np.mean(np.square(audio[i * frame_size:(i + 1) * frame_size]))))
            for i in range(num_frames)
        ]
        # 15th percentile of frame energy estimates the background noise floor
        p15_linear = float(np.percentile(frame_rms, 15))
        noise_floor_db = _amplitude_to_db(p15_linear)
    else:
        noise_floor_db = min(rms_db, -60.0)

    # 4. Signal-to-Noise Ratio (SNR)
    # Estimated as peak level minus quiet background floor
    snr_db = max(0.0, peak_db - noise_floor_db)

    # 5. Silence Detection
    is_silent = peak_db < -50.0

    # 6. Quality Rating and Recommendations
    recs: List[str] = []
    
    if is_silent:
        quality_rating = "CRITICAL"
        recs.append("Microphone signal is silent or extremely quiet (< -50 dBFS).")
        recs.append("Check physical microphone mute switch or increase input gain in Windows Sound Settings.")
    elif has_clipping:
        quality_rating = "POOR"
        recs.append(f"Digital clipping detected ({clipping_ratio * 100:.2f}% of samples). Audio distortion will impair transcription accuracy.")
        recs.append("Lower microphone input volume or move slightly further from the microphone.")
    elif snr_db < 12.0:
        quality_rating = "FAIR"
        recs.append(f"Low Signal-to-Noise Ratio ({snr_db:.1f} dB). Ambient background noise is high.")
        recs.append("Speak closer to the microphone or reduce environmental noise (fans, air conditioning).")
    elif rms_db < -32.0:
        quality_rating = "FAIR"
        recs.append(f"Microphone level is somewhat low (RMS: {rms_db:.1f} dBFS). Optimal speech is between -24 and -14 dBFS.")
        recs.append("Position microphone 6-12 inches from mouth.")
    elif snr_db >= 25.0 and -26.0 <= rms_db <= -12.0 and not has_clipping:
        quality_rating = "EXCELLENT"
        recs.append("Microphone audio is crisp, optimal level, with negligible background noise.")
        recs.append("Hardware is fully verified for accurate clinical documentation.")
    else:
        quality_rating = "GOOD"
        recs.append("Audio levels and noise floor are suitable for Faster-Whisper clinical transcription.")

    return AudioDiagnosticResult(
        device_name=device_name,
        device_index=device_index,
        sample_rate=sample_rate,
        duration_seconds=duration_sec,
        rms_db=rms_db,
        peak_db=peak_db,
        noise_floor_db=noise_floor_db,
        snr_db=snr_db,
        clipping_ratio=clipping_ratio,
        is_silent=is_silent,
        has_clipping=has_clipping,
        quality_rating=quality_rating,
        recommendations=recs,
    )


def run_audio_diagnostic(
    device_index: Optional[int] = None,
    duration: float = 3.0,
    sample_rate: int = 16000
) -> AudioDiagnosticResult:
    """Capture live microphone audio and compute hardware diagnostic report.
    
    Args:
        device_index: Optional PortAudio device index. None selects system default.
        duration: Duration of test recording in seconds.
        sample_rate: Target sample rate (16000 Hz recommended for Faster-Whisper).
        
    Returns:
        AudioDiagnosticResult containing analysis and recommendations.
    """
    import sounddevice as sd

    dev_name = "Default System Mic"
    if device_index is not None:
        try:
            info = sd.query_devices(device_index)
            dev_name = info.get("name", f"Device #{device_index}")
        except Exception:
            dev_name = f"Device #{device_index}"

    samples_to_record = int(duration * sample_rate)
    try:
        recording = sd.rec(
            samples_to_record,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_index
        )
        sd.wait()
    except Exception as exc:
        return AudioDiagnosticResult(
            device_name=dev_name,
            device_index=device_index,
            sample_rate=sample_rate,
            duration_seconds=duration,
            rms_db=-96.0,
            peak_db=-96.0,
            noise_floor_db=-96.0,
            snr_db=0.0,
            clipping_ratio=0.0,
            is_silent=True,
            has_clipping=False,
            quality_rating="CRITICAL",
            recommendations=[f"Hardware capture failed: {exc}. Verify microphone availability."]
        )

    return analyze_audio_data(
        audio=recording,
        sample_rate=sample_rate,
        device_name=dev_name,
        device_index=device_index
    )
