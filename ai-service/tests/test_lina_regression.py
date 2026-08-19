from app.core.nlp.extraction import extract_for_item


def by_code(text, code, label="info"):
    return [x for x in extract_for_item({"text": text, "label": label}) if x.get("code") == code]


def test_lina_maternal_vitals_are_distinct_from_fetal_heart_rate():
    maternal = "فحصنا لها ضغطها طلع 128 على 82 نبضها 78 حرارتها 36.8 ووزنها 72 كيلو"
    fetal = "ارتفاع الرحم 30 سنتيمتر نبض الجنين 148 وهذا الشي طبيعي"

    bp = by_code(maternal, "bp")[0]
    pulse = by_code(maternal, "pulse")[0]
    temp = by_code(maternal, "temp")[0]
    weight = by_code(maternal, "weight_kg")[0]
    fhr = by_code(fetal, "fetal_heart_rate_bpm")[0]
    fundal = by_code(fetal, "fundal_height_cm")[0]

    assert (bp["value"], bp["value2"]) == (128.0, 82.0)
    assert pulse["value"] == 78.0
    assert temp["value"] == 36.8
    assert weight["value"] == 72.0
    assert fhr["value"] == 148.0
    assert fundal["value"] == 30.0
    assert by_code(fetal, "pulse") == []
    # A measured FHR is not automatically a CTG procedure.
    assert by_code(fetal, "ctg") == []


def test_lina_urine_protein_and_glucose_are_structured():
    text = "السكر 92 والبروتين بالبول سلبي"
    glucose = by_code(text, "blood_glucose")[0]
    protein = by_code(text, "urine_protein")[0]
    assert glucose["value"] == 92.0
    assert glucose["unit"] == "mg/dL"
    assert protein["status"] == "normal"
    assert protein["assertion"] == "present"
    assert by_code(text, "urine_protein_test")


def test_lina_normal_fetal_movement_and_cephalic_presentation_are_preserved():
    movement = by_code("حركة الجنين طبيعية ومنتظمة", "fetal_movement_normal")[0]
    presentation = by_code("الجنين بوضعية رأسية وما في انقباضات رحمية", "fetal_presentation")[0]
    contractions = by_code("الجنين بوضعية رأسية وما في انقباضات رحمية", "contractions")[0]
    assert movement["status"] == "normal"
    assert presentation["status"] == "cephalic"
    assert contractions["assertion"] == "absent"
