"""Acoustic speaker diarization and voice clustering for clinical consultations.

Combines digital audio signal processing (pitch F0, spectral centroid, spectral rolloff,
ZCR, RMS energy) with unsupervised 2-speaker clustering and clinical role anchoring
to accurately identify what the Doctor said versus what dialogues the Patient said.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import soundfile as sf

from src.models.patient_record import SpeakerRole, TranscriptSegment
from src.processing.speaker_processor import estimate_speaker_role, process_transcript_segments

logger = logging.getLogger(__name__)


def extract_acoustic_features(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extracts a normalized acoustic voice vector from an audio slice.
    
    Features:
        0: Estimated Pitch F0 (Hz)
        1: Spectral Centroid (Hz)
        2: Spectral Spread / Bandwidth (Hz)
        3: Spectral Rolloff 85% (Hz)
        4: Zero Crossing Rate (ZCR)
        5: Root Mean Square (RMS) Energy
    """
    if len(signal) < 512:
        return np.zeros(6, dtype=np.float32)

    # 1. RMS Energy
    rms = float(np.sqrt(np.mean(signal**2)))

    # 2. Zero Crossing Rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal))))) / 2.0

    # 3. FFT Spectrum & Spectral Descriptors
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    total_energy = float(np.sum(fft)) + 1e-12

    spectral_centroid = float(np.sum(freqs * fft) / total_energy)
    spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft) / total_energy))

    cum_energy = np.cumsum(fft)
    rolloff_idx = np.where(cum_energy >= 0.85 * total_energy)[0]
    spectral_rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])

    # 4. Fundamental Frequency (F0) Pitch via Normalized Autocorrelation (70 Hz - 350 Hz)
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[len(signal) - 1 :]
    min_lag = max(1, int(sr / 350))
    max_lag = min(len(corr) - 1, int(sr / 70))

    if max_lag > min_lag:
        search_window = corr[min_lag:max_lag]
        pitch_lag = min_lag + int(np.argmax(search_window))
        pitch = float(sr / pitch_lag) if pitch_lag > 0 else 0.0
    else:
        pitch = 0.0

    return np.array([pitch, spectral_centroid, spectral_spread, spectral_rolloff, zcr, rms], dtype=np.float32)


def cluster_segments_by_voice(
    features: np.ndarray,
    max_iter: int = 25
) -> np.ndarray:
    """Performs 2-speaker unsupervised K-Means clustering on acoustic feature matrix."""
    n_samples = len(features)
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)

    # Standardize features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    norm_x = (features - mean) / std

    # Initialize centroids: pick two points with greatest Euclidean separation
    c0 = norm_x[0]
    dists = np.linalg.norm(norm_x - c0, axis=1)
    c1 = norm_x[int(np.argmax(dists))]

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        d0 = np.linalg.norm(norm_x - c0, axis=1)
        d1 = np.linalg.norm(norm_x - c1, axis=1)
        new_labels = (d1 < d0).astype(int)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        if np.any(labels == 0):
            c0 = np.mean(norm_x[labels == 0], axis=0)
        if np.any(labels == 1):
            c1 = np.mean(norm_x[labels == 1], axis=0)

    return labels


def diarize_consultation_audio(
    segments: List[TranscriptSegment],
    audio_path: Optional[Union[str, Path]] = None
) -> List[TranscriptSegment]:
    """Annotates transcript segments by fusing acoustic voice clustering and clinical heuristics.
    
    If audio_path is provided and valid, extracts acoustic features per segment and clusters
    into Doctor vs Patient voice profiles. If audio is not available, uses two-pass linguistic
    smoothing.
    
    Returns:
        Annotated List[TranscriptSegment] with speaker_role, role_confidence, and speaker_reason.
    """
    if not segments:
        return []

    # If no audio provided or invalid path, use enhanced linguistic processor
    if not audio_path or not Path(audio_path).is_file():
        return process_transcript_segments(segments)

    try:
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception as e:
        logger.warning(f"Could not read audio file for diarization ({e}); falling back to linguistic processor.")
        return process_transcript_segments(segments)

    total_samples = len(data)
    feature_list = []
    valid_indices = []

    for idx, seg in enumerate(segments):
        start_sample = max(0, int(seg.start_time * sr))
        end_sample = min(total_samples, int(seg.end_time * sr))
        if end_sample > start_sample + 256:
            chunk = data[start_sample:end_sample]
            feats = extract_acoustic_features(chunk, sr)
            feature_list.append(feats)
            valid_indices.append(idx)

    # Need at least 2 valid segments to cluster
    if len(feature_list) < 2:
        return process_transcript_segments(segments)

    feature_matrix = np.array(feature_list)
    cluster_labels = cluster_segments_by_voice(feature_matrix)

    # Map cluster assignments back to segments
    seg_clusters: dict[int, int] = {}
    for pos, seg_idx in enumerate(valid_indices):
        seg_clusters[seg_idx] = int(cluster_labels[pos])

    # First pass: Get linguistic anchor evidence for both clusters
    c0_doc_evidence = 0.0
    c0_pt_evidence = 0.0
    c1_doc_evidence = 0.0
    c1_pt_evidence = 0.0

    for idx, seg in enumerate(segments):
        role, conf, _ = estimate_speaker_role(seg.text)
        cluster = seg_clusters.get(idx)
        if cluster is not None and conf >= 0.50:
            if cluster == 0:
                if role == SpeakerRole.DOCTOR:
                    c0_doc_evidence += conf
                elif role == SpeakerRole.PATIENT:
                    c0_pt_evidence += conf
            elif cluster == 1:
                if role == SpeakerRole.DOCTOR:
                    c1_doc_evidence += conf
                elif role == SpeakerRole.PATIENT:
                    c1_pt_evidence += conf

    # Determine which cluster corresponds to Doctor vs Patient
    # Case A: Cluster 0 has stronger doctor evidence or Cluster 1 has stronger patient evidence
    if (c0_doc_evidence + c1_pt_evidence) >= (c1_doc_evidence + c0_pt_evidence):
        cluster_role_map = {0: SpeakerRole.DOCTOR, 1: SpeakerRole.PATIENT}
    else:
        cluster_role_map = {1: SpeakerRole.DOCTOR, 0: SpeakerRole.PATIENT}

    # Annotate segments with fused acoustic + linguistic results
    for idx, seg in enumerate(segments):
        cluster = seg_clusters.get(idx)
        ling_role, ling_conf, ling_reason = estimate_speaker_role(seg.text)

        if cluster is not None:
            acoustic_role = cluster_role_map[cluster]
            if ling_role == acoustic_role and ling_role != SpeakerRole.UNKNOWN:
                seg.speaker_role = acoustic_role
                seg.role_confidence = min(0.95, max(ling_conf + 0.15, 0.85))
                seg.speaker_reason = f"Acoustic voice clustering & linguistic agreement ({ling_reason})"
            elif ling_conf >= 0.80:
                # Strong explicit linguistic anchor takes precedence
                seg.speaker_role = ling_role
                seg.role_confidence = ling_conf
                seg.speaker_reason = ling_reason
            else:
                # Use acoustic voice cluster to resolve short / conversational utterances
                seg.speaker_role = acoustic_role
                seg.role_confidence = 0.75
                seg.speaker_reason = f"Acoustic voice cluster {cluster} matches {acoustic_role.value.lower()}"
        else:
            seg.speaker_role = ling_role
            seg.role_confidence = ling_conf
            seg.speaker_reason = ling_reason

    # Final pass: Contextual continuity smoothing to catch any remaining unassigned segments
    return process_transcript_segments(segments)
