from app.core.nlp.canonicalization import CanonicalizationCandidate
from app.core.pipeline import MedicalScribePipeline


class FakeTranscriber:
    model_size = "fake-whisper"

    def transcribe(self, _path):
        return {
            "language": "ar",
            "segments": [{
                "start": 0.0,
                "end": 2.0,
                "text": "في عندها وجع راس خفيف",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.01,
                "compression_ratio": 1.0,
            }],
        }


class FakeClassifier:
    max_len = 128
    model_name = "fake-arabert"

    def __init__(self):
        self.seen = []

    def predict_batch(self, texts):
        self.seen = list(texts)
        return [("symptom", 0.95) for _ in texts]


class FakeCanonicalizer:
    applied = True
    model_name = "fake-qwen-shadow"

    def canonicalize_batch(self, texts):
        assert texts == ["في عندها وجع راس خفيف"]
        return [CanonicalizationCandidate("تعاني من صداع خفيف", 0.92)]


def test_shadow_mode_stores_candidate_but_clinical_path_uses_raw_text():
    classifier = FakeClassifier()
    pipeline = MedicalScribePipeline(
        FakeTranscriber(), classifier,
        canonicalizer=FakeCanonicalizer(),
        canonicalization_mode="shadow",
        uncertainty_enabled=False,
    )
    report = pipeline.process("fake.wav", "job-shadow")
    assert classifier.seen == ["في عندها وجع راس خفيف"]
    item = report.soap["subjective"].items[0]
    assert item.text == "في عندها وجع راس خفيف"
    assert item.text_raw == "في عندها وجع راس خفيف"
    assert item.text_canonical == "تعاني من صداع خفيف"
    assert item.canonicalization_status == "accepted"
    assert report.pipeline_meta.canonicalization_mode == "shadow"
    assert report.pipeline_meta.canonicalization_accepted == 1


def test_active_mode_exists_only_for_explicit_benchmark_comparison():
    classifier = FakeClassifier()
    pipeline = MedicalScribePipeline(
        FakeTranscriber(), classifier,
        canonicalizer=FakeCanonicalizer(),
        canonicalization_mode="active",
        uncertainty_enabled=False,
    )
    report = pipeline.process("fake.wav", "job-active")
    assert classifier.seen == ["تعاني من صداع خفيف"]
    assert report.soap["subjective"].items[0].text == "تعاني من صداع خفيف"
    assert report.pipeline_meta.canonicalization_mode == "active"
