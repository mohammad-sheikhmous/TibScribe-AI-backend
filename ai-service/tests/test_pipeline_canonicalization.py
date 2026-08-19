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
    model_name = "fake-arat5"

    def canonicalize_batch(self, texts):
        assert texts == ["في عندها وجع راس خفيف"]
        return [CanonicalizationCandidate("تعاني من صداع خفيف", 0.92)]


def test_pipeline_classifies_canonical_text_but_keeps_raw_asr_and_audio_traceability():
    classifier = FakeClassifier()
    pipeline = MedicalScribePipeline(
        FakeTranscriber(), classifier,
        canonicalizer=FakeCanonicalizer(),
        uncertainty_enabled=False,
    )
    report = pipeline.process("fake.wav", "job-canon")
    assert classifier.seen == ["تعاني من صداع خفيف"]
    item = report.soap["subjective"].items[0]
    assert item.text == "تعاني من صداع خفيف"
    assert item.text_raw == "في عندها وجع راس خفيف"
    assert item.text_canonical == "تعاني من صداع خفيف"
    assert item.canonicalization_status == "accepted"
    assert item.start_sec == 0.0 and item.end_sec == 2.0
    assert "تعاني من صداع خفيف" in report.soap_formatted["subjective"].text
    assert report.pipeline_meta.canonicalization_accepted == 1
