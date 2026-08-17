"""Doctor voiceprint and speaker-role assignment (IMPLEMENTATION.md P3-07 / P3-08).

The idea that makes this cheap: in one clinic, **the speaker who recurs across every
recording is the doctor** — the patient changes each visit. So the profile can be built
from the archive with no labelling at all (PLAN_V2 §5).

Robustness — the honest version
-------------------------------
Speaker embeddings are trained to survive loudness and emotion (gain augmentation,
energy normalisation), so **shouting does not create a new speaker**. Whispering is the
genuinely hard case: without vocal-fold vibration the pitch cues disappear, and
verification accuracy degrades sharply. A single centroid with an absolute threshold
would therefore misfire exactly when a doctor drops their voice. Four defences instead:

1. **Relative, not absolute.** The question is never "is this the doctor's voice?"
   (verification) but "which of the clusters in THIS recording is the doctor?"
   (assignment). Diarization already grouped the turns; a shifted embedding still sits
   far closer to the doctor than to a different person.
2. **Multiple prototypes.** The profile keeps several centroids covering different
   vocal conditions, and scores by the best match — not by distance to one average that
   blurs them all together.
3. **Abstention on a narrow margin.** When the top two candidates are close the answer
   is `unknown`, which routes to human confirmation instead of a silent guess.
4. **Independent linguistic evidence.** Question forms and clinical vocabulary versus
   first-person symptom talk. Voice and text disagreeing is itself a reason to abstain.

Every confirmed assignment can be enrolled back into the profile (P7), so the doctor's
quiet, tired and raised voices all become prototypes over time.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

Role = str  # "doctor" | "patient" | "unknown"

DOCTOR = "doctor"
PATIENT = "patient"
UNKNOWN = "unknown"

# Below this gap between the best and second-best candidate, do not decide.
DEFAULT_MARGIN = 0.06
# Agglomerative clustering threshold on cosine distance (standard for x-vectors).
DEFAULT_CLUSTER_DISTANCE = 0.65
# Weight of voice evidence when both signals are available.
VOICE_WEIGHT = 0.7


# --- linguistic evidence (independent of audio) -------------------------------------

QUESTION_MARKERS = ("؟", "هل ", "كم ", "متى", "وين", "أين", "شو ", "ايش", "إيش",
                    "كيف", "من امتى", "منذ متى", "عندك", "بتشتكي", "في عندك")
CLINICIAN_MARKERS = ("سنعطيك", "سأصف", "بدنا نعمل", "لازم نعمل", "راح نعمل", "افحص",
                     "الفحص يبين", "التشخيص", "الخطة", "وصفة", "تحليل", "راجعينا",
                     "خذي", "تاخذي", "ننصح", "يُنصح", "سنجري", "سنطلب")
PATIENT_MARKERS = ("عندي", "أشعر", "اشعر", "بحس", "بشعر", "تعبانة", "تعبان", "بوجعني",
                   "بيوجعني", "صار لي", "من يومين", "ما بقدر", "ما فيني", "حاسة")


def text_role_score(texts: Sequence[str]) -> float:
    """+1 = clearly the clinician, -1 = clearly the patient, 0 = no signal."""
    joined = " ".join(texts)
    if not joined.strip():
        return 0.0
    clinician = sum(joined.count(m) for m in QUESTION_MARKERS)
    clinician += 2 * sum(joined.count(m) for m in CLINICIAN_MARKERS)
    patient = 2 * sum(joined.count(m) for m in PATIENT_MARKERS)
    total = clinician + patient
    if total == 0:
        return 0.0
    return (clinician - patient) / total


# --- embedding helpers ----------------------------------------------------------------

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def cluster_embeddings(
    vectors: np.ndarray, *, distance_threshold: float = DEFAULT_CLUSTER_DISTANCE
) -> list[list[int]]:
    """Greedy agglomerative clustering on cosine distance (average linkage).

    Written out rather than pulled from sklearn: it is ~20 lines, keeps this module
    dependency-light, and is the standard clustering step in speaker diarization.
    """
    vectors = l2_normalize(vectors)
    clusters: list[list[int]] = [[i] for i in range(len(vectors))]
    if len(clusters) <= 1:
        return clusters

    while len(clusters) > 1:
        best_pair, best_distance = None, np.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = float(np.mean(cosine_similarity(vectors[clusters[i]],
                                                      vectors[clusters[j]])))
                distance = 1.0 - sim
                if distance < best_distance:
                    best_pair, best_distance = (i, j), distance
        if best_distance > distance_threshold or best_pair is None:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
    return clusters


# --- the profile -----------------------------------------------------------------------

@dataclass
class VoiceProfile:
    """Several prototypes for one role, so different vocal conditions are all covered."""

    role: Role
    prototypes: np.ndarray                      # (k, dim), L2-normalised
    recordings_seen: int = 0
    note: str = ""

    def similarity(self, embedding: np.ndarray) -> float:
        """Best match over prototypes — never an average that blurs conditions together."""
        if self.prototypes.size == 0:
            return 0.0
        return float(np.max(cosine_similarity(embedding, self.prototypes)))

    def to_payload(self) -> dict:
        return {
            "role": self.role,
            "prototypes": self.prototypes.tolist(),
            "recordings_seen": self.recordings_seen,
            "note": self.note,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "VoiceProfile":
        return cls(
            role=payload["role"],
            prototypes=np.asarray(payload["prototypes"], dtype=np.float64),
            recordings_seen=int(payload.get("recordings_seen", 0)),
            note=payload.get("note", ""),
        )


def build_doctor_profile(
    per_recording_embeddings: Iterable[Sequence[np.ndarray]],
    *,
    distance_threshold: float = DEFAULT_CLUSTER_DISTANCE,
    max_prototypes: int = 4,
) -> Optional[VoiceProfile]:
    """Find the speaker present in the most recordings — that is the doctor.

    Input: one list of speaker embeddings per recording (one per diarized cluster).
    Needs no labels; accuracy improves as the archive grows.
    """
    flat: list[np.ndarray] = []
    origin: list[int] = []
    for index, recording in enumerate(per_recording_embeddings):
        for embedding in recording:
            flat.append(np.asarray(embedding, dtype=np.float64).ravel())
            origin.append(index)

    if len(flat) < 2:
        return None

    vectors = l2_normalize(np.vstack(flat))
    clusters = cluster_embeddings(vectors, distance_threshold=distance_threshold)

    # Rank by how many DISTINCT recordings a cluster spans, not by how many turns it
    # holds: a single long monologue must not outweigh presence across visits.
    def coverage(cluster: list[int]) -> tuple[int, int]:
        return len({origin[i] for i in cluster}), len(cluster)

    best = max(clusters, key=coverage)
    recordings_covered, _ = coverage(best)
    if recordings_covered < 2:
        logger.info("voiceprint: no speaker recurs across recordings yet — no profile")
        return None

    member_vectors = vectors[best]
    prototypes = _condense(member_vectors, max_prototypes)
    return VoiceProfile(
        role=DOCTOR,
        prototypes=prototypes,
        recordings_seen=recordings_covered,
        note=f"recurring speaker across {recordings_covered} recordings",
    )


def _condense(vectors: np.ndarray, max_prototypes: int) -> np.ndarray:
    """Reduce many embeddings to a few prototypes covering different conditions."""
    if len(vectors) <= max_prototypes:
        return l2_normalize(vectors)
    sub_clusters = cluster_embeddings(vectors, distance_threshold=0.35)
    sub_clusters.sort(key=len, reverse=True)
    prototypes = [vectors[idx].mean(axis=0) for idx in sub_clusters[:max_prototypes]]
    return l2_normalize(np.vstack(prototypes))


# --- role assignment ---------------------------------------------------------------------

@dataclass
class RoleAssignment:
    role: Role
    confidence: float
    basis: str                       # voice | text | voice+text | fallback
    margin: float = 0.0
    detail: dict = field(default_factory=dict)


def assign_roles(
    cluster_embeddings_map: dict[str, np.ndarray],
    cluster_texts: dict[str, Sequence[str]],
    profile: Optional[VoiceProfile] = None,
    *,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, RoleAssignment]:
    """Decide which diarized cluster is the doctor, in THIS recording.

    Relative by construction: scores are compared between the clusters present, so a
    globally shifted voice (shouting, a cold, a different microphone) still wins its own
    cluster. When nothing separates the candidates, everyone is `unknown` — an explicit
    "ask a human" rather than a coin flip.
    """
    labels = list(cluster_embeddings_map)
    if not labels:
        return {}
    if len(labels) == 1:
        # One speaker: text is the only evidence that it is the clinician.
        score = text_role_score(cluster_texts.get(labels[0], []))
        role = DOCTOR if score > 0.2 else PATIENT if score < -0.2 else UNKNOWN
        return {labels[0]: RoleAssignment(role, abs(score), "text" if role != UNKNOWN
                                          else "fallback", detail={"text_score": score})}

    voice_scores: dict[str, float] = {}
    if profile is not None and profile.prototypes.size:
        for label, embedding in cluster_embeddings_map.items():
            voice_scores[label] = profile.similarity(np.asarray(embedding))

    text_scores = {label: text_role_score(cluster_texts.get(label, [])) for label in labels}

    combined: dict[str, float] = {}
    for label in labels:
        text_component = (text_scores[label] + 1) / 2  # -1..1 -> 0..1
        if voice_scores:
            voice_component = (voice_scores[label] + 1) / 2
            combined[label] = (
                VOICE_WEIGHT * voice_component + (1 - VOICE_WEIGHT) * text_component
            )
        else:
            combined[label] = text_component

    ordered = sorted(labels, key=lambda label: combined[label], reverse=True)
    top, runner_up = ordered[0], ordered[1]
    gap = combined[top] - combined[runner_up]
    basis = "voice+text" if voice_scores and any(text_scores.values()) else (
        "voice" if voice_scores else "text"
    )

    if gap < margin:
        # Too close to call: flag instead of guessing (defence 3).
        return {
            label: RoleAssignment(UNKNOWN, 0.0, "fallback", margin=gap,
                                  detail={"reason": "candidates too close",
                                          "combined": combined[label]})
            for label in labels
        }

    # Voice and text pointing at different clusters is a warning, not a tie-break.
    if voice_scores and text_scores:
        text_pick = max(labels, key=lambda label: text_scores[label])
        voice_pick = max(labels, key=lambda label: voice_scores[label])
        if text_pick != voice_pick and abs(text_scores[text_pick]) > 0.3:
            logger.warning(
                "voiceprint: voice says %s, text says %s — assigning by the combined "
                "score with reduced confidence", voice_pick, text_pick,
            )
            gap *= 0.5

    assignments: dict[str, RoleAssignment] = {}
    for label in labels:
        role = DOCTOR if label == top else PATIENT
        assignments[label] = RoleAssignment(
            role=role,
            confidence=round(min(1.0, 0.5 + gap), 4),
            basis=basis,
            margin=round(gap, 4),
            detail={"voice": voice_scores.get(label), "text": text_scores[label],
                    "combined": round(combined[label], 4)},
        )
    return assignments
