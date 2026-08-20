from app.core.nlp.discourse import enrich_cross_segment_context
from app.core.nlp.extraction import extract_for_item
from app.core.report.builder import build_report
from app.core.report.schema import ClassifiedSegment


def seg(i: int, text: str, label: str, *, confidence: float = 0.9) -> ClassifiedSegment:
    s = ClassifiedSegment(
        order_index=i,
        text=text,
        start_sec=float(i),
        end_sec=float(i + 1),
        source_segment_index=i,
        label=label,
        confidence=confidence,
        asr_confidence=0.9,
    )
    s.entity_links = extract_for_item({"text": text, "label": label})
    return s


def find_link(segment: ClassifiedSegment, code: str) -> dict:
    return next(link for link in (segment.entity_links or []) if link.get("code") == code)


def test_contraction_frequency_and_strength_inherit_previous_labour_anchor():
    parts = enrich_cross_segment_context([
        seg(0, "اجت بسبب تقلصات منتظمة من حوالي 4 ساعات", "symptom"),
        seg(1, "وتقول ان الوجه عم يجي تقريبا كل 5 دقايق وصار اقوى من قبل", "symptom"),
    ])
    interval = find_link(parts[1], "contraction_interval_min")
    stronger = find_link(parts[1], "contractions_strengthened")
    assert interval["value"] == 5.0 and interval["unit"] == "min"


def test_adjacent_fluid_leak_plus_suspicion_is_not_confirmed_rom():
    parts = enrich_cross_segment_context([
        seg(0, "تقلصات منتظمة", "symptom"),
        seg(1, "لاحظت نزول سائل", "symptom"),
        seg(2, "وبتشك انه هو مي الجنين وحركة الجنين موجودة", "diagnosis"),
    ])
    leak = find_link(parts[1], "vaginal_fluid_leak")
    rom = find_link(parts[2], "water_breaking")
    assert leak["assertion"] == "present"
    assert rom["assertion"] == "present"
    assert rom["status"] == "suspected"


def test_exam_and_cervical_dilation_route_to_objective_and_keep_structured_value():
    parts = enrich_cross_segment_context([
        seg(0, "بالفحص كانت العلامات الحيوية مستقرة", "diagnosis"),
        seg(1, "وعنق الرحم متوسعة تقريبا 5 سم", "diagnosis"),
    ])
    report = build_report("labour-job", parts)
    objective_text = [x.text for x in report.soap["objective"].items]
    assert "بالفحص كانت العلامات الحيوية مستقرة" in objective_text
    assert "وعنق الرحم متوسعة تقريبا 5 سم" in objective_text
    cervix = find_link(parts[1], "cervical_dilation_cm")
    assert cervix["value"] == 5.0 and cervix["unit"] == "cm"


def test_monitoring_and_conditional_scope_stay_in_plan_across_segments():
    parts = enrich_cross_segment_context([
        seg(0, "لذلك رح نكمل مراقبة تقدم المخاط", "plan"),
        seg(1, "ونبض الجنين وحالة الام بشكل مستمر", "pregnancy_risk", confidence=0.25),
        seg(2, "واذا صار اي تغيير بنبض الجنين", "pregnancy_risk"),
        seg(3, "او توقف بتقدم المخاط", "diagnosis"),
        seg(4, "وقتها لازم نعيد تقييم الخطة من اول وجديد", "plan"),
    ])
    report = build_report("labour-job", parts)
    plan_text = [x.text for x in report.soap["plan"].items]
    assert plan_text == [s.text for s in parts]
    assert find_link(parts[1], "plan_continuation")["kind"] == "context"
    assert find_link(parts[1], "ctg")["assertion"] == "planned"
    assert find_link(parts[2], "ctg")["assertion"] == "hypothetical"
    assert find_link(parts[3], "contingency_condition")["kind"] == "context"


def test_nonmenstrual_heavy_bleeding_phrase_falls_back_to_vaginal_bleeding():
    s = seg(0, "هلا مافي نزيف غزير بس لاحظت نزول سائل", "symptom")
    codes = {link["code"] for link in s.entity_links or []}
    assert "heavy_menstrual_bleeding" not in codes
    assert "vaginal_bleeding" in codes
    bleeding = find_link(s, "vaginal_bleeding")
    assert bleeding["assertion"] == "absent"

def test_lina_danger_sign_list_inherits_conditional_scope():

    parts = enrich_cross_segment_context([
        seg(
            14,
            "اذا حستت فيها مثل انه يصير فيها "
            "عندها وجع راس اوي يصير عندها",
            "symptom",
        ),
        seg(
            15,
            "نزيف تشوش بالرؤية او حست بحركة "
            "الجنين ببطنها انه هي خفت اذا",
            "symptom",
        ),
        seg(
            16,
            "صار في احد هاي الاعراض فهي لازم "
            "تراجعني حتى لو كان قبل اسبوعين",
            "diagnosis",
        ),
    ])

    for code in (
        "vaginal_bleeding",
        "blurred_vision",
        "reduced_fetal_movement",
    ):
        assert (
            find_link(
                parts[1],
                code,
            )["assertion"]
            == "hypothetical"
        )

    assert (
        find_link(
            parts[1],
            "contingency_condition",
        )["kind"]
        == "context"
    )

    assert (
        find_link(
            parts[2],
            "contingency_action",
        )["kind"]
        == "context"
    )