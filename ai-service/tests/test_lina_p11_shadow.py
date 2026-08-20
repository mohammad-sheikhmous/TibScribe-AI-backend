from app.core.nlp.canonicalization import CanonicalizationCandidate
from app.core.pipeline import MedicalScribePipeline

RAW = [
    "مريضة لينا محمد عمرها 29 سنة حامل بالاسبوع 32",
    "هيتاني حمل لإلها الولادة السابقة كانت طبيعية كانت قبلت 30 سنين بدون مضاعفات",
    "في عندها وجع راس خفيف كل يومين رجليها شوي متنفخين",
    "ما عندها لا تشوش بالرؤية ولا نزيف ولا قلم بالبطن",
    "بالنسبة لحركة الجنين فهي طبيعية منتظمة",
    "فحصنا لها ضغطها طلع 128 على 82 نبضها 78 حرارتها 36.8 ووزنها 72 كيلو",
    "ارتفاع الرحم 30 سنتيمتر نبض الجنين 148 وهذا الشي طبيعي",
    "الجنين اخذ وضعية رئيسية وما عنده انقباضات رحمية",
    "عملنا كم تحليل وطلعت نتيجة الهيموغلوبين كان 11.5",
    "السكر 92 والبروتين طلع عندها بالبول سلبي",
    "اما بالنسبة لوظائف الكبد والكلافة هي طبيعية",
    "وصفت لها حديد وحمض فوليك اسيت نصحتها انها ترتاح وتقلل ملح وتراقب الضغط",
    "يعني تروح وتجي تقيسه طلبت انها تعمل صنار للنمو والسائل الامينوسي بالاسبوع الجاي",
    "موعد موعدنا الجاي بعد اسبوعين شرحت لها شو الشغلات اذ هي بتشكل خطر",
    "اذا حستت فيها مثل انه يصير فيها عندها وجع راس اوي يصير عندها",
    "نزيف تشوش بالرؤية او حست بحركة الجنين ببطنها انه هي خفت اذا",
    "صار في احد هاي الاعراض فهي لازم تراجعني حتى لو كان قبل اسبوعين",
]

# Deliberately include the known bad P10-style question transformations. P11 must store
# or reject them without letting them alter the classifier/KBS decision source.
CANDIDATES = [
    RAW[0],
    "إنه حمل لها الولادة السابقة كانت طبيعية كانت قبلت 30 سنوات بدون مضاعفات",
    "هل لديها وجع راس خفيف كل يومين رجليها قليلا متنفخين",
    RAW[3],
    RAW[4],
    "لقد فحصنا لها ضغطها خرج 128 على 82 نبضها 78 حرارتها 36.8 ووزنها 72 كيلو",
    "إن ارتفاع الرحم 30 سنتيمتر نبض الجنين 148 وهذا الشيء طبيعي",
    "لقد أخذ الجنين وضعية رئيسية وليس لديه انقباضات رحمية",
    "لقد قمنا بعدة تحليل وخرجت نتيجة الهيموغلوبين كان 11.5",
    "هل السكر 92 والبروتين خرج لديها في البول سلبي",
    "أما بالنسبة لوظائف الكبد والكلافة فهي طبيعية",
    "لقد وصفت لها حديد وحمض فوليك اسيت نصحتها بأن ترتاح وتقلل ملح وتراقب الضغط",
    RAW[12], RAW[13], RAW[14], RAW[15], RAW[16],
]

LABELS = [
    "info", "history", "symptom", "history", "vital", "vital", "vital", "diagnosis",
    "lab", "lab", "lab", "medication", "plan", "plan", "symptom", "symptom", "diagnosis",
]


class LinaTranscriber:
    model_size = "large-v3"

    def transcribe(self, _path):
        return {
            "language": "ar",
            "segments": [
                {
                    "start": float(i * 5), "end": float(i * 5 + 4.5), "text": text,
                    "avg_logprob": -0.13, "no_speech_prob": 0.01, "compression_ratio": 1.0,
                }
                for i, text in enumerate(RAW)
            ],
        }


class LinaClassifier:
    max_len = 128
    model_name = "fake-arabert-lina"

    def __init__(self):
        self.seen = []

    def predict_batch(self, texts):
        self.seen = list(texts)
        return [(label, 0.9) for label in LABELS]


class BadP10Canonicalizer:
    applied = True
    model_name = "bad-p10-candidate"

    def canonicalize_batch(self, texts):
        assert list(texts) == RAW
        return [CanonicalizationCandidate(text, 0.8) for text in CANDIDATES]


def test_lina_shadow_mode_cannot_regress_classifier_or_entities_and_formats_structured_facts():
    classifier = LinaClassifier()
    report = MedicalScribePipeline(
        LinaTranscriber(), classifier,
        canonicalizer=BadP10Canonicalizer(), canonicalization_mode="shadow",
        uncertainty_enabled=False,
    ).process("lina.wav", "lina-p11")

    assert classifier.seen == RAW
    assert report.pipeline_meta.canonicalization_mode == "shadow"
    # The two known statement->question rewrites are rejected by the P11 safety gate.
    by_order = {
        item.order_index: item
        for section in report.soap.values()
        for item in section.items
    }
    assert by_order[2].canonicalization_status == "rejected"
    assert "speech_act_changed" in by_order[2].canonicalization_reasons
    assert by_order[9].canonicalization_status == "rejected"

    # Clinical path stays on RAW and therefore retains the original label assignment.
    assert by_order[2].label == "symptom"
    assert by_order[14].label == "symptom"
    assert by_order[14].label != "postpartum"

    # Entity improvements from P10/P11 remain active on the raw decision text.
    vitals = {e["code"]: e for e in by_order[5].entity_links}
    assert vitals["bp"]["value"] == 128.0
    assert vitals["pulse"]["value"] == 78.0
    assert vitals["weight_kg"]["value"] == 72.0
    fetal = {e["code"]: e for e in by_order[6].entity_links}
    assert fetal["fetal_heart_rate_bpm"]["value"] == 148.0
    assert fetal["fundal_height_cm"]["value"] == 30.0

    objective = report.soap_formatted["objective"].text
    assert "ضغط الدم 128/82 مم زئبق" in objective
    assert "نبض الجنين 148 ض/د" in objective
    assert "الهيموغلوبين 11.5 غ/دل" in objective
    assert "سكر الدم 92 ملغ/دل" in objective
    assert "بروتين البول سلبي" in objective

    labs = {e["code"]: e for e in by_order[9].entity_links}
    assert labs["urine_protein"]["status"] == "normal"

    plan = report.soap_formatted["plan"].text
    assert "الحديد" in plan and "حمض الفوليك" in plan
    assert "تقليل الملح" in plan
    assert "مراقبة ضغط الدم في المنزل" in plan
    assert "سونار لتقييم نمو الجنين والسائل الأمنيوسي خلال الأسبوع القادم" in plan
    assert "موعد المتابعة بعد أسبوعين" in plan
    assert "علامات الخطر التي تستدعي المراجعة الفورية" in plan
    assert "اذا حستت فيها" not in plan

    warnings = report.soap_formatted["subjective"].warnings
    assert any("العمر 29" in warning and "قبل 30 سنة" in warning for warning in warnings)
