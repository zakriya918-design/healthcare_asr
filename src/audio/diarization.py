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
import re
from src.models.patient_record import SpeakerRole, TranscriptSegment
from src.processing.speaker_processor import (
    estimate_speaker_role,
    process_transcript_segments,
    disentangle_and_heal_transcript_segments,
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
                        if 0 < lag < len(corr) - 1:
                            alpha = float(corr[lag - 1])
                            beta = float(corr[lag])
                            gamma = float(corr[lag + 1])
                            denom = 2.0 * (2.0 * beta - alpha - gamma)
                            if abs(denom) > 1e-6:
                                delta = (alpha - gamma) / denom
                                refined_lag = lag + delta
                            else:
                                refined_lag = float(lag)
                        else:
                            refined_lag = float(lag)
                        if refined_lag > 0:
                            pitches.append(sr / refined_lag)

    pitch_mean = float(np.mean(pitches)) if pitches else 0.0
    pitch_std = float(np.std(pitches)) if len(pitches) > 1 else 0.0
    pitch_median = float(np.median(pitches)) if pitches else 0.0
    voiced_ratio = float(len(pitches)) / max(1, len(signal) // hop_len)

    return np.array([
        pitch_mean, spectral_centroid, spectral_spread, rolloff85, zcr, rms,
        rolloff50, b1, b2, b3, b4, pitch_std, pitch_median, voiced_ratio
    ], dtype=np.float32)


def compute_cluster_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Computes mean silhouette score to evaluate clustering quality with vectorized speed."""
    unique_labels = np.unique(labels)
    n_samples = len(X)
    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        return -1.0

    # Vectorized pairwise Euclidean distance matrix
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)

    sil_values = []
    for i in range(n_samples):
        current_label = labels[i]
        same_mask = (labels == current_label)
        same_mask[i] = False
        if np.any(same_mask):
            a = float(np.mean(dists[i, same_mask]))
        else:
            a = 0.0

        b = np.inf
        for other_label in unique_labels:
            if other_label == current_label:
                continue
            other_mask = (labels == other_label)
            if np.any(other_mask):
                b = min(b, float(np.mean(dists[i, other_mask])))

        denom = max(a, b)
        s = (b - a) / denom if denom > 0 else 0.0
        sil_values.append(s)

    return float(np.mean(sil_values))


def score_cluster_conversational_profile(texts: List[str]) -> Dict[SpeakerRole, float]:
    """Computes generalized, specialty-agnostic clinical role likelihoods from universal conversational dynamics.
    
    Uses universal dialogue markers across all medical encounters:
    - Doctor: 2nd-person address ('you', 'your'), clinical inquiries (questions), prescriptive directives, greeting patient with honorifics.
    - Patient: 1st-person singular ('I', 'my', 'me'), somatic complaint descriptors (pain, discomfort, onset), addressing doctor with honorifics.
    - Caregiver: 3rd-person relational pronouns ('he', 'she', 'my mother', 'my husband', 'my child', 'his', 'her').
    - Nurse: Triage & vitals intake ('vitals', 'blood pressure', 'temperature', 'scale', 'doctor will be in').
    """
    total_words = 0
    cnt_1st = 0
    cnt_2nd = 0
    cnt_3rd = 0
    cnt_questions = 0
    cnt_directives = 0
    cnt_symptoms = 0
    cnt_address_doc = 0
    cnt_address_pt = 0
    cnt_triage = 0

    re_1st = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|myself)\b", re.IGNORECASE)
    re_2nd = re.compile(r"\b(?:you|you're|you've|you'd|you'll|your|yours|yourself)\b", re.IGNORECASE)
    re_3rd = re.compile(r"\b(?:he|he's|him|his|she|she's|her|hers|they|them|their|my\s+(?:mother|father|husband|wife|son|daughter|child|mom|dad|parent))\b", re.IGNORECASE)
    
    re_directive = re.compile(
        r"\b(?:order|prescribe|recommend|suggest|examine|listen\s+to|check\s+your|take\s+this|avoid|schedule|follow\s*up|come\s+back|see\s+me|return\s+to|let's\s+look|let\s+me\s+listen|let\s+me\s+examine)\b",
        re.IGNORECASE
    )
    re_symptom = re.compile(
        r"\b(?:pain|hurt|hurts|aching|sore|discomfort|bothering|started|noticed|feeling|numb|swelling|tired|fatigue|cough|fever|trouble|hard\s+to|can't\s+sleep|struggling|headache|headaches|fell|confused)\b",
        re.IGNORECASE
    )
    # Genuine vocative addressing the doctor (e.g. "Doctor, I...", "thank you, doctor", "good morning, doctor")
    re_address_doc = re.compile(
        r"(?:,\s*(?:dr\.?|doctor)\b|\bthank\s+you\s*,?\s*(?:dr\.?|doctor)\b|\bhello\s*,?\s*(?:dr\.?|doctor)\b|\bgood\s+morning\s*,?\s*(?:dr\.?|doctor)\b|\byes\s*,?\s*(?:dr\.?|doctor)\b|^\s*(?:dr\.?|doctor)\s*,\s*(?:i\b|my\b|we\b|he\b|she\b))",
        re.IGNORECASE
    )
    re_address_pt = re.compile(
        r"\b(?:mr\.|mrs\.|ms\.|miss)\s+[A-Za-z]+\b",
        re.IGNORECASE
    )
    re_triage = re.compile(
        r"\b(?:vitals|blood\s+pressure|temperature|scale|pulse\s+oximeter|doctor\s+will\s+be\s+in|room\s+you\s+in)\b",
        re.IGNORECASE
    )

    for text in texts:
        words = text.split()
        total_words += len(words)
        cnt_1st += len(re_1st.findall(text))
        cnt_2nd += len(re_2nd.findall(text))
        cnt_3rd += len(re_3rd.findall(text))
        if "?" in text or re.match(r"^\s*(?:how|what|when|where|why|do\s+you|have\s+you|did\s+you|are\s+you|is\s+there|any)\b", text, re.IGNORECASE):
            cnt_questions += 1
        cnt_directives += len(re_directive.findall(text))
        cnt_symptoms += len(re_symptom.findall(text))
        if re_address_doc.search(text):
            cnt_address_doc += 1
        if re_address_pt.search(text):
            cnt_address_pt += 1
        if re_triage.search(text):
            cnt_triage += 1

    n_utts = max(1, len(texts))
    norm = max(1.0, total_words / 10.0)

    doc_score = (
        (cnt_2nd / norm) * 2.5 +
        (cnt_questions / n_utts) * 3.5 +
        (cnt_directives / norm) * 3.5 +
        (cnt_address_pt * 5.0) -
        (cnt_address_doc * 6.0) -
        (cnt_3rd / norm) * 2.0
    )

    pt_score = (
        (cnt_1st / norm) * 2.5 +
        (cnt_symptoms / norm) * 2.5 +
        (cnt_address_doc * 3.0) -
        (cnt_3rd / norm) * 4.0 -
        (cnt_directives / norm) * 3.0 -
        (cnt_address_pt * 5.0)
    )

    if cnt_3rd > 0:
        cg_score = (
            (cnt_3rd / norm) * 5.0 +
            (cnt_symptoms / norm) * 1.5 +
            (cnt_address_doc * 2.0) -
            (cnt_address_pt * 5.0) -
            (cnt_directives / norm) * 3.0
        )
    else:
        cg_score = 0.01

    nurse_score = (
        (cnt_triage * 4.0) +
        (cnt_2nd / norm) * 1.2 +
        (cnt_questions / n_utts) * 1.5
    )

    return {
        SpeakerRole.DOCTOR: max(0.01, float(doc_score)),
        SpeakerRole.PATIENT: max(0.01, float(pt_score)),
        SpeakerRole.CAREGIVER: max(0.01, float(cg_score)),
        SpeakerRole.NURSE: max(0.01, float(nurse_score)),
    }


FEATURE_WEIGHTS = np.array([
    2.5,   # 0: Pitch F0 mean
    2.0,   # 1: Spectral Centroid
    1.5,   # 2: Spectral Spread
    1.0,   # 3: Spectral Rolloff 85%
    1.0,   # 4: ZCR
    0.15,  # 5: RMS Energy (attenuate volume level variations)
    1.0,   # 6: Spectral Rolloff 50%
    1.8,   # 7: Subband 1 (70-300 Hz)
    1.8,   # 8: Subband 2 (300-1000 Hz)
    1.8,   # 9: Subband 3 (1000-3000 Hz)
    0.3,   # 10: Subband 4 (high-frequency hiss)
    1.2,   # 11: Pitch std
    2.5,   # 12: Pitch F0 median
    1.0,   # 13: Voiced ratio
], dtype=np.float32)


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

    # Standardize features with perceptual timbre & pitch weighting
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    norm_x = ((features - mean) / std) * FEATURE_WEIGHTS

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

    # Disentangle composite multi-speaker turns and heal severed mid-sentence fragments
    segments = disentangle_and_heal_transcript_segments(segments)

    # If no audio provided or invalid path, return healed and contextually annotated segments
    if not audio_path or not Path(audio_path).is_file():
        return segments

    try:
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception as e:
        logger.warning(f"Could not read audio file for diarization ({e}); falling back to linguistic processor.")
        return segments

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
        return segments

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

    # Tally generalized conversational dynamics profile for each detected speaker cluster
    cluster_evidence: Dict[int, Dict[SpeakerRole, float]] = {}
    for c in unique_clusters:
        c_texts = [segments[idx].text for idx in valid_indices if seg_clusters.get(idx) == c]
        cluster_evidence[c] = score_cluster_conversational_profile(c_texts)
        # Augment with individual high-confidence anchor matches
        for idx in valid_indices:
            if seg_clusters.get(idx) == c:
                role, conf, _ = estimate_speaker_role(segments[idx].text)
                if role in cluster_evidence[c] and conf >= 0.70:
                    cluster_evidence[c][role] += conf * 3.0

    # Global optimal assignment of unique clusters to distinct roles
    k_clusters = len(unique_clusters)
    if k_clusters == 2:
        has_cg = any(cluster_evidence[c][SpeakerRole.CAREGIVER] > 2.0 for c in unique_clusters)
        has_nurse = any(cluster_evidence[c][SpeakerRole.NURSE] > 2.0 for c in unique_clusters)
        if not has_cg and not has_nurse:
            role_candidates = [SpeakerRole.DOCTOR, SpeakerRole.PATIENT]
        elif has_cg and not has_nurse:
            role_candidates = [SpeakerRole.DOCTOR, SpeakerRole.PATIENT, SpeakerRole.CAREGIVER]
        elif has_nurse and not has_cg:
            role_candidates = [SpeakerRole.DOCTOR, SpeakerRole.PATIENT, SpeakerRole.NURSE]
        else:
            role_candidates = [SpeakerRole.DOCTOR, SpeakerRole.PATIENT, SpeakerRole.CAREGIVER, SpeakerRole.NURSE]
    else:
        role_candidates = [SpeakerRole.DOCTOR, SpeakerRole.PATIENT, SpeakerRole.CAREGIVER, SpeakerRole.NURSE]

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

    # Annotate all segments with multi-modal acoustic + linguistic fusion
    for idx, seg in enumerate(segments):
        cluster = seg_clusters.get(idx)
        ling_role, ling_conf, ling_reason = estimate_speaker_role(seg.text)

        if cluster is not None:
            seg.speaker_id = f"Speaker {cluster + 1}"
            acoustic_role = cluster_role_map[cluster]

            # Decisive clinical phrasing or explicit script prefix takes precedence
            if ling_conf >= 0.60 and ling_role != SpeakerRole.UNKNOWN:
                seg.speaker_role = ling_role
                seg.role_confidence = max(ling_conf, 0.88)
                seg.speaker_reason = f"Linguistic {ling_role.value.lower()} phrasing ({ling_reason})"
            else:
                # Acoustic voice cluster role for brief or ambiguous utterances
                seg.speaker_role = acoustic_role
                seg.role_confidence = 0.80
                seg.speaker_reason = f"Acoustic {seg.speaker_id} voice cluster mapped to {acoustic_role.value.lower()}"
        else:
            if ling_role != SpeakerRole.UNKNOWN:
                seg.speaker_role = ling_role
                seg.role_confidence = ling_conf
                seg.speaker_reason = ling_reason

    # Final contextual pass to resolve any lingering short utterances
    segments = process_transcript_segments(segments)

    # Harmonize physical speaker_id to aligned clinical role if unassigned
    for seg in segments:
        if not seg.speaker_id or seg.speaker_id == "Speaker":
            if seg.speaker_role == SpeakerRole.DOCTOR:
                seg.speaker_id = "Speaker 1"
            elif seg.speaker_role == SpeakerRole.PATIENT:
                seg.speaker_id = "Speaker 2"
            elif seg.speaker_role == SpeakerRole.CAREGIVER:
                seg.speaker_id = "Speaker 3"
            elif seg.speaker_role == SpeakerRole.NURSE:
                seg.speaker_id = "Speaker 4"

    return segments
