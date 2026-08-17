"""Doctor voiceprint + role assignment (IMPLEMENTATION.md P3-07 / P3-08).

Synthetic embeddings rather than real audio: the questions under test are about the
DECISION LOGIC, and a real encoder would only make them slower and non-deterministic.

The scenarios are the ones raised in review:
  * the doctor shouts  -> embedding shifts moderately
  * the doctor whispers -> embedding shifts a lot (the genuinely hard case)
  * voice and language evidence disagree
  * the two speakers are indistinguishable
"""
import numpy as np
import pytest

from app.core.asr.voiceprint import (
    DOCTOR,
    PATIENT,
    UNKNOWN,
    VoiceProfile,
    assign_roles,
    build_doctor_profile,
    cluster_embeddings,
    text_role_score,
)

# 192 dims to match a real ECAPA-TDNN embedding. It matters: in a low-dimensional
# space two unrelated speakers land within chance distance of each other often enough
# to make clustering thresholds meaningless (an early 32-dim version of this test
# merged everyone into one cluster for exactly that reason).
DIM = 192


def voice(seed: int) -> np.ndarray:
    """A stable identity vector for a fictional speaker."""
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=DIM)
    return vector / np.linalg.norm(vector)


def shifted(base: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Same speaker under a different vocal condition: identity + a perturbation.

    amount≈0.15 stands in for raised/lowered volume, ≈0.45 for whispering.
    """
    generator = np.random.default_rng(seed)
    noise = generator.normal(size=DIM)
    noise /= np.linalg.norm(noise)
    shifted_vector = base + amount * noise
    return shifted_vector / np.linalg.norm(shifted_vector)


DOCTOR_VOICE = voice(1)
PATIENT_A = voice(2)
PATIENT_B = voice(3)
PATIENT_C = voice(4)

DOCTOR_TEXT = ["هل عندك صداع؟", "من امتى بتشتكي؟", "سنعطيك مسكن وتحليل بول"]
PATIENT_TEXT = ["عندي صداع من يومين", "بحس بدوخة وما فيني أقوم"]


# --- linguistic evidence --------------------------------------------------------------

def test_text_score_separates_clinician_from_patient():
    assert text_role_score(DOCTOR_TEXT) > 0.3
    assert text_role_score(PATIENT_TEXT) < -0.3
    assert text_role_score(["الطقس جميل اليوم"]) == 0.0


# --- clustering --------------------------------------------------------------------------

def test_clustering_groups_one_speakers_variations_together():
    vectors = np.vstack([
        DOCTOR_VOICE, shifted(DOCTOR_VOICE, 0.15, 1), shifted(DOCTOR_VOICE, 0.2, 2),
        PATIENT_A, shifted(PATIENT_A, 0.15, 3),
    ])
    clusters = cluster_embeddings(vectors)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 3]


# --- profile building (unsupervised) -------------------------------------------------

def test_profile_picks_the_speaker_who_recurs_across_recordings():
    recordings = [
        [shifted(DOCTOR_VOICE, 0.12, 10), PATIENT_A],
        [shifted(DOCTOR_VOICE, 0.15, 11), PATIENT_B],
        [shifted(DOCTOR_VOICE, 0.10, 12), PATIENT_C],
    ]
    profile = build_doctor_profile(recordings)

    assert profile is not None
    assert profile.recordings_seen == 3
    # The profile matches the doctor far better than any patient.
    assert profile.similarity(DOCTOR_VOICE) > 0.8
    assert profile.similarity(PATIENT_A) < 0.5


def test_no_profile_from_a_single_recording():
    """One visit cannot tell which speaker is the recurring one."""
    assert build_doctor_profile([[DOCTOR_VOICE, PATIENT_A]]) is None


def test_profile_keeps_several_prototypes_for_different_conditions():
    """A shout and a whisper must both be represented, not averaged into mush."""
    recordings = [
        [DOCTOR_VOICE, PATIENT_A],
        [shifted(DOCTOR_VOICE, 0.45, 21), PATIENT_B],   # whisper-like
        [shifted(DOCTOR_VOICE, 0.20, 22), PATIENT_C],   # raised voice
        [shifted(DOCTOR_VOICE, 0.05, 23), voice(9)],
    ]
    profile = build_doctor_profile(recordings, distance_threshold=0.75)
    assert profile is not None and len(profile.prototypes) >= 2


# --- the review scenarios ---------------------------------------------------------------

def _assign(doctor_embedding, patient_embedding, profile=None, **kwargs):
    return assign_roles(
        {"A": doctor_embedding, "B": patient_embedding},
        {"A": DOCTOR_TEXT, "B": PATIENT_TEXT},
        profile,
        **kwargs,
    )


def test_normal_conditions_assign_correctly():
    profile = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]))
    result = _assign(DOCTOR_VOICE, PATIENT_A, profile)
    assert result["A"].role == DOCTOR and result["B"].role == PATIENT


def test_shouting_does_not_create_a_new_speaker():
    """The review question: a raised voice shifts the embedding but not the decision."""
    profile = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]))
    result = _assign(shifted(DOCTOR_VOICE, 0.25, 30), PATIENT_A, profile)
    assert result["A"].role == DOCTOR
    assert result["A"].basis in ("voice", "voice+text")


def test_whispering_is_rescued_by_the_relative_comparison():
    """The hard case: a large shift would fail an absolute threshold, yet the shifted
    voice is still closer to the doctor's prototypes than the patient is."""
    profile = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]))
    whispered = shifted(DOCTOR_VOICE, 0.45, 31)
    assert profile.similarity(whispered) < 0.95          # an absolute test would wobble
    assert profile.similarity(whispered) > profile.similarity(PATIENT_A)
    result = _assign(whispered, PATIENT_A, profile)
    assert result["A"].role == DOCTOR


def test_enrolling_the_whisper_makes_it_a_prototype():
    """Feedback (P7) closes the loop: once confirmed, the quiet voice is known."""
    whispered = shifted(DOCTOR_VOICE, 0.45, 32)
    single = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]))
    enrolled = VoiceProfile(role=DOCTOR, prototypes=np.vstack([DOCTOR_VOICE, whispered]))
    assert enrolled.similarity(whispered) > single.similarity(whispered)


def test_indistinguishable_speakers_abstain_instead_of_guessing():
    profile = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]))
    twin = shifted(DOCTOR_VOICE, 0.02, 33)
    result = assign_roles({"A": DOCTOR_VOICE, "B": twin},
                          {"A": [], "B": []}, profile)  # no linguistic signal either
    assert {r.role for r in result.values()} == {UNKNOWN}


def test_text_alone_works_before_any_profile_exists():
    """Day one: no archive yet, so language carries the decision."""
    result = _assign(DOCTOR_VOICE, PATIENT_A, None)
    assert result["A"].role == DOCTOR and result["B"].role == PATIENT
    assert result["A"].basis == "text"


def test_conflicting_evidence_lowers_confidence():
    """Voice says A, language says B: decide, but with a visibly weaker score."""
    profile = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]))
    agreeing = assign_roles({"A": DOCTOR_VOICE, "B": PATIENT_A},
                            {"A": DOCTOR_TEXT, "B": PATIENT_TEXT}, profile)
    conflicting = assign_roles({"A": DOCTOR_VOICE, "B": PATIENT_A},
                               {"A": PATIENT_TEXT, "B": DOCTOR_TEXT}, profile)
    assert conflicting["A"].confidence < agreeing["A"].confidence


def test_single_speaker_recording_is_labelled_from_language():
    result = assign_roles({"A": DOCTOR_VOICE}, {"A": DOCTOR_TEXT}, None)
    assert result["A"].role == DOCTOR


def test_three_speakers_yield_one_doctor():
    profile = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]))
    result = assign_roles(
        {"A": DOCTOR_VOICE, "B": PATIENT_A, "C": PATIENT_B},
        {"A": DOCTOR_TEXT, "B": PATIENT_TEXT, "C": ["أنا أمها"]},
        profile,
    )
    assert sum(1 for r in result.values() if r.role == DOCTOR) == 1


def test_profile_round_trips_through_storage():
    profile = VoiceProfile(role=DOCTOR, prototypes=np.array([DOCTOR_VOICE]),
                           recordings_seen=5, note="x")
    restored = VoiceProfile.from_payload(profile.to_payload())
    assert restored.recordings_seen == 5
    assert np.allclose(restored.prototypes, profile.prototypes)
