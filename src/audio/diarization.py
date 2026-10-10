"""Acoustic multi-speaker diarization and voice clustering for clinical consultations.

Supports multi-person consultation rooms (Doctor, Patient, Caregiver/Family, Nurse/Staff).
Combines:
- 14-dimensional acoustic voiceprint representation (F0 pitch mean/std/median, spectral centroid,
  spectral spread, spectral rolloff 85% & 50%, 4-band spectral energy, zero-crossing rate, RMS energy)
- Dynamic K-means clustering with automatic speaker count detection (K=2, 3, 4) using silhouette scoring
- Multi-role clinical anchor mapping (Doctor, Patient, Caregiver, Nurse)
- Per-segment physical speaker tagging ("Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4")
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf

import itertools
from src.models.patient_record import SpeakerRole, TranscriptSegment
from src.processing.speaker_processor import (
    estimate_speaker_role,
    process_transcript_segments,
    PREFIX_DOCTOR_RE,
    PREFIX_PATIENT_RE,
    PREFIX_CAREGIVER_RE,
    PREFIX_NURSE_RE,
)

logger = logging.getLogger(__name__)


def extract_acoustic_features(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extracts a comprehensive 14-dimensional acoustic voice vector from an audio slice.
    
    Features:
        0: Estimated Pitch F0 (Hz)
        1: Spectral Centroid (Hz)
        2: Spectral Spread / Bandwidth (Hz)
        3: Spectral Rolloff 85% (Hz)
        4: Zero Crossing Rate (ZCR)
        5: Root Mean Square (RMS) Energy
        6: Spectral Rolloff 50% (Hz)
        7: Subband 1 Energy (70-300 Hz)
        8: Subband 2 Energy (300-1000 Hz)
        9: Subband 3 Energy (1000-3000 Hz)
        10: Subband 4 Energy (3000-8000 Hz)
        11: Pitch F0 Std Dev (Hz)
        12: Pitch F0 Median (Hz)
        13: Voiced Frame Ratio (0.0 to 1.0)
    """
    if len(signal) < 512:
        return np.zeros(14, dtype=np.float32)

    # 1. Energy & Zero Crossing Rate
    rms = float(np.sqrt(np.mean(signal**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(signal))))) / 2.0

    # 2. FFT Spectrum & Spectral Descriptors
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    total_energy = float(np.sum(fft)) + 1e-12

    spectral_centroid = float(np.sum(freqs * fft) / total_energy)
    spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft) / total_energy))

    cum_energy = np.cumsum(fft)
    idx85 = np.where(cum_energy >= 0.85 * total_energy)[0]
    idx50 = np.where(cum_energy >= 0.50 * total_energy)[0]
    rolloff85 = float(freqs[idx85[0]]) if len(idx85) > 0 else float(freqs[-1])
    rolloff50 = float(freqs[idx50[0]]) if len(idx50) > 0 else float(freqs[-1])

    # 3. Energy Subbands (Formant Regions)
    b1 = float(np.sum(fft[(freqs >= 70) & (freqs < 300)])) / total_energy
    b2 = float(np.sum(fft[(freqs >= 300) & (freqs < 1000)])) / total_energy
    b3 = float(np.sum(fft[(freqs >= 1000) & (freqs < 3000)])) / total_energy
    b4 = float(np.sum(fft[(freqs >= 3000) & (freqs < 8000)])) / total_energy

    # 4. Fundamental Frequency (F0) Tracking
    frame_len = min(len(signal), max(512, int(sr * 0.04)))  # 40ms frame
    hop_len = max(256, int(sr * 0.02))  # 20ms hop
    pitches = []
    min_lag = max(1, int(sr / 350))
    max_lag = min(frame_len - 1, int(sr / 70))

    if max_lag > min_lag:
        for i in range(0, len(signal) - frame_len + 1, hop_len):
            frame = signal[i : i + frame_len]
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(frame) - 1 :]
            if max_lag < len(corr):
                search_win = corr[min_lag:max_lag]
                if len(search_win) > 0:
                    peak_idx = int(np.argmax(search_win))
                    lag = min_lag + peak_idx
                    if lag > 0 and corr[lag] > 0.3 * corr[0]:
                        pitches.append(sr / lag)

    pitch_mean = float(np.mean(pitches)) if pitches else 0.0
    pitch_std = float(np.std(pitches)) if len(pitches) > 1 else 0.0
    pitch_median = float(np.median(pitches)) if pitches else 0.0
    voiced_ratio = float(len(pitches)) / max(1, len(signal) // hop_len)

    return np.array([
        pitch_mean, spectral_centroid, spectral_spread, rolloff85, zcr, rms,
        rolloff50, b1, b2, b3, b4, pitch_std, pitch_median, voiced_ratio
    ], dtype=np.float32)


def compute_cluster_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Computes mean silhouette score to evaluate clustering quality."""
    unique_labels = np.unique(labels)
    n_samples = len(X)
    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        return -1.0

    sil_values = []
    for i in range(n_samples):
        same_mask = labels == labels[i]
        same_cluster = X[same_mask]
        if len(same_cluster) > 1:
            a = np.mean([np.linalg.norm(X[i] - p) for p in same_cluster if not np.array_equal(X[i], p)])
        else:
            a = 0.0

        b = np.inf
        for other_label in unique_labels:
            if other_label == labels[i]:
                continue
            diff_cluster = X[labels == other_label]
            if len(diff_cluster) > 0:
                dist = np.mean([np.linalg.norm(X[i] - p) for p in diff_cluster])
                b = min(b, dist)

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        sil_values.append(s)

    return float(np.mean(sil_values))


def cluster_segments_by_voice(
    features: np.ndarray,
    num_speakers: Union[int, str] = "auto",
    max_k: int = 4,
    max_iter: int = 30
) -> np.ndarray:
    """Performs multi-speaker voice clustering (K=2 to 4) on acoustic feature matrix."""
    n_samples = len(features)
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)

    # Standardize features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    norm_x = (features - mean) / std

    # Determine candidate cluster counts
    if isinstance(num_speakers, int) and num_speakers >= 2:
        candidate_k = [min(num_speakers, n_samples)]
    else:
        max_possible_k = min(max_k, n_samples, 4)
        candidate_k = list(range(2, max_possible_k + 1)) if max_possible_k >= 2 else [1]

    best_labels = np.zeros(n_samples, dtype=int)
    best_score = -2.0

    for k in candidate_k:
        if k == 1:
            return np.zeros(n_samples, dtype=int)

        k_best_labels = None
        k_best_inertia = np.inf

        # Multiple restarts with K-means++ initialization
        for _ in range(5):
            rng = np.random.default_rng(42 + _)
            centroids = [norm_x[rng.integers(n_samples)]]
            for _c in range(1, k):
                dists_sq = np.min([np.linalg.norm(norm_x - c, axis=1)**2 for c in centroids], axis=0)
                probs = dists_sq / (np.sum(dists_sq) + 1e-12)
                next_c = norm_x[int(np.argmax(probs))]
                centroids.append(next_c)
            centroids_arr = np.array(centroids)

            labels = np.zeros(n_samples, dtype=int)
            for _it in range(max_iter):
                dists = np.array([np.linalg.norm(norm_x - c, axis=1) for c in centroids_arr])
                new_labels = np.argmin(dists, axis=0)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for j in range(k):
                    if np.any(labels == j):
                        centroids_arr[j] = np.mean(norm_x[labels == j], axis=0)

            inertia = sum(
                np.sum(np.linalg.norm(norm_x[labels == j] - centroids_arr[j], axis=1)**2)
                for j in range(k) if np.any(labels == j)
            )
            if inertia < k_best_inertia:
                k_best_inertia = inertia
                k_best_labels = labels

        score = compute_cluster_silhouette(norm_x, k_best_labels)
        if score > best_score:
            best_score = score
            best_labels = k_best_labels

    return best_labels


def diarize_consultation_audio(
    segments: List[TranscriptSegment],
    audio_path: Optional[Union[str, Path]] = None,
    num_speakers: Union[int, str] = "auto"
) -> List[TranscriptSegment]:
    """Annotates transcript segments for multi-speaker rooms (2, 3, or 4 persons).
    
    Combines:
    - Acoustic voice clustering: assigns speaker_id ("Speaker 1", "Speaker 2", etc.)
    - Multi-role clinical anchor mapping: maps acoustic clusters to Doctor, Patient, Caregiver, or Nurse.
    - Continuity smoothing: resolves brief conversational utterances.
    
    Returns:
        Annotated List[TranscriptSegment] with speaker_id, speaker_role, and confidence.
    """
    if not segments:
        return []

    # If no audio provided or invalid path, use enhanced linguistic processor
    if not audio_path or not Path(audio_path).is_file():
        processed = process_transcript_segments(segments)
        for idx, seg in enumerate(processed):
            if not seg.speaker_id:
                seg.speaker_id = seg.speaker_role.value if seg.speaker_role != SpeakerRole.UNKNOWN else f"Speaker {(idx % 2) + 1}"
        return processed

    try:
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception as e:
        logger.warning(f"Could not read audio file for diarization ({e}); falling back to linguistic processor.")
        processed = process_transcript_segments(segments)
        for idx, seg in enumerate(processed):
            if not seg.speaker_id:
                seg.speaker_id = seg.speaker_role.value if seg.speaker_role != SpeakerRole.UNKNOWN else f"Speaker {(idx % 2) + 1}"
        return processed

    total_samples = len(data)
    feature_list = []
    valid_indices = []

    for idx, seg in enumerate(segments):
        start_sample = max(0, int(seg.start_time * sr)) if seg.start_time is not None else 0
        end_sample = min(total_samples, int(seg.end_time * sr)) if seg.end_time is not None else 0
        if end_sample > start_sample + 256:
            chunk = data[start_sample:end_sample]
            feats = extract_acoustic_features(chunk, sr)
            feature_list.append(feats)
            valid_indices.append(idx)

    # Need at least 2 valid segments to cluster
    if len(feature_list) < 2:
        processed = process_transcript_segments(segments)
        for idx, seg in enumerate(processed):
            if not seg.speaker_id:
                seg.speaker_id = seg.speaker_role.value if seg.speaker_role != SpeakerRole.UNKNOWN else f"Speaker {(idx % 2) + 1}"
        return processed

    feature_matrix = np.array(feature_list)
    cluster_labels = cluster_segments_by_voice(feature_matrix, num_speakers=num_speakers)
    unique_clusters = list(sorted(np.unique(cluster_labels)))

    # Micro-cluster pruning: If K >= 3 and some clusters are tiny outliers (< 8% of segments),
    # reassign them to the closest major centroid to prevent spurious speaker tags in 2-person dialogues
    if len(unique_clusters) >= 3:
        counts = {c: int(np.sum(cluster_labels == c)) for c in unique_clusters}
        major_clusters = [c for c in unique_clusters if counts[c] >= max(3, int(0.08 * len(cluster_labels)))]
        if len(major_clusters) >= 2 and len(major_clusters) < len(unique_clusters):
            centroids = {c: np.mean(feature_matrix[cluster_labels == c], axis=0) for c in major_clusters}
            for i in range(len(cluster_labels)):
                if cluster_labels[i] not in major_clusters:
                    dists = {mc: float(np.linalg.norm(feature_matrix[i] - centroids[mc])) for mc in major_clusters}
                    cluster_labels[i] = min(dists.keys(), key=lambda mc: dists[mc])
            unique_clusters = list(sorted(np.unique(cluster_labels)))

    # Map cluster assignments back to segments
    seg_clusters: Dict[int, int] = {}
    for pos, seg_idx in enumerate(valid_indices):
        seg_clusters[seg_idx] = int(cluster_labels[pos])

    # Tally clinical evidence for each detected speaker cluster
    cluster_evidence: Dict[int, Dict[SpeakerRole, float]] = {
        c: {
            SpeakerRole.DOCTOR: 0.0,
            SpeakerRole.PATIENT: 0.0,
            SpeakerRole.CAREGIVER: 0.0,
            SpeakerRole.NURSE: 0.0
        }
        for c in unique_clusters
    }

    for idx, seg in enumerate(segments):
        role, conf, _ = estimate_speaker_role(seg.text)
        cluster = seg_clusters.get(idx)
        if cluster is not None and role in cluster_evidence[cluster] and conf >= 0.40:
            cluster_evidence[cluster][role] += conf

    # Global optimal assignment of unique clusters to distinct roles
    role_candidates = [SpeakerRole.DOCTOR, SpeakerRole.PATIENT, SpeakerRole.CAREGIVER, SpeakerRole.NURSE]
    k_clusters = len(unique_clusters)
    best_mapping = {}
    best_score = -1.0

    for perm in itertools.permutations(role_candidates, k_clusters):
        current_score = sum(cluster_evidence[c][role] for c, role in zip(unique_clusters, perm))
        if current_score > best_score:
            best_score = current_score
            best_mapping = {c: role for c, role in zip(unique_clusters, perm)}

    cluster_role_map = best_mapping

    # Ensure Doctor and Patient are prioritized if unassigned
    assigned_roles = set(cluster_role_map.values())
    for c in unique_clusters:
        if c not in cluster_role_map:
            if SpeakerRole.DOCTOR not in assigned_roles:
                cluster_role_map[c] = SpeakerRole.DOCTOR
                assigned_roles.add(SpeakerRole.DOCTOR)
            elif SpeakerRole.PATIENT not in assigned_roles:
                cluster_role_map[c] = SpeakerRole.PATIENT
                assigned_roles.add(SpeakerRole.PATIENT)
            elif SpeakerRole.CAREGIVER not in assigned_roles:
                cluster_role_map[c] = SpeakerRole.CAREGIVER
                assigned_roles.add(SpeakerRole.CAREGIVER)
            else:
                cluster_role_map[c] = SpeakerRole.NURSE

    # Annotate all segments consistently with physical speaker_id and assigned acoustic role
    for idx, seg in enumerate(segments):
        cluster = seg_clusters.get(idx)
        ling_role, ling_conf, ling_reason = estimate_speaker_role(seg.text)

        if cluster is not None:
            seg.speaker_id = f"Speaker {cluster + 1}"
            acoustic_role = cluster_role_map[cluster]

            # Explicit prefixes typed by user ("Doctor:", "Patient:", etc.) take highest precedence
            if ling_conf >= 0.95 and (PREFIX_DOCTOR_RE.match(seg.text) or PREFIX_PATIENT_RE.match(seg.text) or PREFIX_CAREGIVER_RE.match(seg.text) or PREFIX_NURSE_RE.match(seg.text)):
                seg.speaker_role = ling_role
                seg.role_confidence = ling_conf
                seg.speaker_reason = ling_reason
            else:
                # Acoustic voice cluster is the invariant ground truth for the speaker
                seg.speaker_role = acoustic_role
                seg.role_confidence = 0.88
                seg.speaker_reason = f"Acoustic {seg.speaker_id} voice cluster mapped to {acoustic_role.value.lower()}"
        else:
            seg.speaker_id = ling_role.value if ling_role != SpeakerRole.UNKNOWN else "Speaker"
            seg.speaker_role = ling_role
            seg.role_confidence = ling_conf
            seg.speaker_reason = ling_reason

    return segments
