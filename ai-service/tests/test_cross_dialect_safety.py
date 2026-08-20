import pytest

from app.core.nlp.extraction import extract_for_item


@pytest.mark.parametrize("text", [
    "عندها وجع راس خفيف وما عندها نزيف",             # Levantine
    "عندها صداع خفيف ومفيش نزيف",                    # Egyptian
    "عندها صداع خفيف وما في نزيف",                   # Gulf/Levantine
    "عدها صداع خفيف وماكو نزف",                      # Iraqi
    "عندها صداع خفيف وما عندهاش نزيف",               # North African-style negation
])
def test_common_dialects_preserve_headache_and_negated_bleeding(text):
    entities = extract_for_item({"text": text, "label": "symptom"})
    by_code = {e["code"]: e for e in entities if e.get("code") in {"headache", "vaginal_bleeding"}}
    assert by_code["headache"]["assertion"] == "present"
    assert by_code["vaginal_bleeding"]["assertion"] == "absent"


def test_levatine_edema_asr_surface_is_extracted_without_rewrite():
    text = "في عندها وجع راس خفيف كل يومين رجليها شوي متنفخين"
    entities = extract_for_item({"text": text, "label": "symptom"})
    codes = {e["code"] for e in entities}
    assert "headache" in codes
    assert "edema" in codes


def test_urine_protein_negative_is_a_lab_result_not_dietary_protein():
    text = "السكر 92 والبروتين طلع عندها بالبول سلبي"
    entities = extract_for_item({"text": text, "label": "lab"})
    by_code = {e["code"]: e for e in entities}
    assert by_code["blood_glucose"]["value"] == 92.0
    assert by_code["urine_protein"]["status"] == "normal"
    assert "protein_rich_food" not in by_code


def test_dietary_protein_is_not_misread_as_urine_protein():
    text = "نصحتها تاكل بروتين أكثر"
    entities = extract_for_item({"text": text, "label": "plan"})
    codes = {e["code"] for e in entities}
    assert "protein_rich_food" in codes
    assert "urine_protein" not in codes
