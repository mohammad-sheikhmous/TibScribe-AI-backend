from app.core.nlp.canonicalization import CanonicalizationCandidate
from app.core.pipeline import MedicalScribePipeline

RAW = [
    "المريضة حامل بالاسبوع 34",
    "عندها صداع قوي وتشوش بالرؤية",
    "ضغطها 160 على 110 والنبض 96 والحرارة 37.2",
    "البروتين بالبول إيجابي",
]
LABELS = ["info", "symptom", "vital", "lab"]


class Transcriber:
    model_size = "large-v3"

    def transcribe(self, _path):
        return {
            "language": "ar",
            "segments": [
                {
                    "start": i * 4.0,
                    "end": i * 4.0 + 3.5,
                    "text": text,
                    "avg_logprob": -0.10,
                    "no_speech_prob": 0.01,
                    "compression_ratio": 1.0,
                }
                for i, text in enumerate(RAW)
            ],
        }


class Classifier:
    max_len = 128
    model_name = "fake-arabert-high-risk"

    def __init__(self):
        self.seen = []

    def predict_batch(self, texts):
        self.seen = list(texts)
        return [(label, 0.95) for label in LABELS]


class UnsafeShadowCorrector:
    applied = True
    model_name = "unsafe-shadow-test"

    def canonicalize_batch(self, texts):
        assert list(texts) == RAW
        return [
            CanonicalizationCandidate(RAW[0]),
            # Statement -> question must be rejected.
            CanonicalizationCandidate("هل لديها صداع قوي وتشوش بالرؤية؟"),
            # Changed BP must be rejected.
            CanonicalizationCandidate("ضغطها 150 على 100 والنبض 96 والحرارة 37.2"),
            # Positive -> negative qualifier must be rejected.
            CanonicalizationCandidate("البروتين بالبول سلبي"),
        ]


def test_high_risk_facts_are_immune_to_bad_shadow_rewrites():
    classifier = Classifier()
    report = MedicalScribePipeline(
        Transcriber(),
        classifier,
        canonicalizer=UnsafeShadowCorrector(),
        canonicalization_mode="shadow",
        uncertainty_enabled=False,
    ).process("high-risk.wav", "p11-high-risk")

    # Non-negotiable P11 invariant: clinical decision path sees the raw ASR text.
    assert classifier.seen == RAW

    by_order = {
        item.order_index: item
        for section in report.soap.values()
        for item in section.items
    }
    assert by_order[1].canonicalization_status == "rejected"
    assert "speech_act_changed" in by_order[1].canonicalization_reasons
    assert by_order[2].canonicalization_status == "rejected"
    assert "numeric_facts_changed" in by_order[2].canonicalization_reasons
    assert by_order[3].canonicalization_status == "rejected"

    symptoms = {e["code"]: e for e in by_order[1].entity_links}
    assert symptoms["headache"]["assertion"] == "present"
    assert symptoms["blurred_vision"]["assertion"] == "present"

    bp = next(e for e in by_order[2].entity_links if e["code"] == "bp")
    assert (bp["value"], bp["value2"]) == (160.0, 110.0)

    protein = next(e for e in by_order[3].entity_links if e["code"] == "urine_protein")
    assert protein["status"] == "high"

    assert "ضغط الدم 160/110 مم زئبق" in report.soap_formatted["objective"].text
    assert "بروتين البول إيجابي" in report.soap_formatted["objective"].text
