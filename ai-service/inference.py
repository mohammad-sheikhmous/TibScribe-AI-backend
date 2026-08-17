"""Standalone classifier smoke-test CLI.

Loads the fine-tuned classifier from model_output/ and prints the predicted label,
confidence, Arabic section name, and SOAP section for each input sentence. This is
a quick manual check of the classifier only — the full audio→report pipeline lives
in the FastAPI service (app/main.py).

Usage:
    python inference.py                      # classify a few built-in OB/GYN samples
    python inference.py "جملة طبية" "جملة أخرى"   # classify sentences from the CLI
"""
from __future__ import annotations

import os
import sys

# Allow `python inference.py` from the repo root to import the shared app package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.core.nlp.classifier import MedicalSentenceClassifier  # noqa: E402
from app.core.nlp.sections import label_ar, soap_for_label  # noqa: E402

SAMPLES = [
    "حاسة بحركة الجنين خفت كتير اليوم وصارلها كم ساعة ما تحركت",
    "بتتحسس من البنسلين وبيعملها طفح جلدي وحكة وضيق نفس شديد",
    "تحليل HbA1c طلع مرتفع بشكل واضح يدل على سوء ضبط السكري",
    "نوصي بإيقاف الميتفورمين مؤقتاً ومراقبة السكر يومياً",
    "المريضة في الأسبوع 34 مع علامات تسمم حمل وارتفاع ضغط",
]


def main(argv: list[str]) -> None:
    sentences = argv or SAMPLES
    clf = MedicalSentenceClassifier()
    print("\n" + "=" * 70)
    for text in sentences:
        label, confidence = clf.predict(text)
        print(f"[{label}] ({confidence:.1%}) → {label_ar(label)} / {soap_for_label(label)}")
        print(f"    {text}")
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1:])
