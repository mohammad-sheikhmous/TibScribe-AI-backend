"""Six-way assertion classification (IMPLEMENTATION.md P5-01 / P5-02).

The regression suite for gaps ك-٢ and ك-٣ — the two largest sources of false alarms
measured in the analysis. Every case here is a sentence shape that actually occurs in
an OB/GYN consultation, not a constructed puzzle.
"""
import pytest

from app.core.nlp.assertion import classify_assertion, find_negation_aware, is_negated


def at(text: str, term: str):
    """Classify `term` where it appears in `text`."""
    start = text.index(term)
    return classify_assertion(text, start, start + len(term))


# --- present: the only actionable category ------------------------------------------

@pytest.mark.parametrize("text,term", [
    ("تشتكي من صداع شديد منذ يومين", "صداع"),
    ("المريضة عندها تورم بالقدمين", "تورم"),
    ("الفحص يبين التهاب بالحوض", "التهاب"),
])
def test_plain_findings_are_present_and_actionable(text, term):
    result = at(text, term)
    assert result.assertion == "present"
    assert result.is_actionable


# --- absent (gap ك-٢) -----------------------------------------------------------------

@pytest.mark.parametrize("text,term", [
    ("لا يوجد نزيف مهبلي", "نزيف"),
    ("صداع لا يوجد عند المريضة", "صداع"),              # negation AFTER the term
    ("المريضة ما عندها تورم", "تورم"),
    ("نفت وجود نزيف", "نزيف"),
    ("تحليل البول سلبي للزلال", "زلال"),
])
def test_denied_findings_are_absent(text, term):
    result = at(text, term)
    assert result.assertion == "absent"
    assert not result.is_actionable


def test_negation_scope_stops_at_a_clause_boundary():
    text = "لا يوجد نزيف لكن يوجد صداع شديد"
    assert at(text, "نزيف").assertion == "absent"
    assert at(text, "صداع").assertion == "present"


def test_pseudo_negation_does_not_negate():
    assert at("لا شك أن لديها صداع نصفي", "صداع").assertion == "present"


# --- hypothetical (gap ك-٣) -------------------------------------------------------------

@pytest.mark.parametrize("text,term", [
    ("إذا صار عندك صداع أو تشوش بالرؤية راجعينا فوراً", "صداع"),
    ("إذا صار عندك صداع أو تشوش بالرؤية راجعينا فوراً", "تشوش بالرؤية"),
    ("في حال حصل نزيف توجهي للطوارئ", "نزيف"),
    ("لو حسيتي بتقلصات قوية اتصلي فينا", "تقلصات"),
    ("راقبي حركة الجنين وإذا قلت راجعينا", "حركة الجنين"),
])
def test_safety_netting_advice_is_hypothetical(text, term):
    """The single most common false-alarm shape: advice about what has NOT happened."""
    result = at(text, term)
    assert result.assertion == "hypothetical"
    assert not result.is_actionable


def test_the_measured_false_alarm_sentence_no_longer_asserts():
    """Verbatim from the analysis report, where it fired a pre-eclampsia alert."""
    text = "إذا صار عندك صداع أو تشوش بالرؤية راجعينا فوراً"
    for term in ("صداع", "تشوش بالرؤية"):
        assert not at(text, term).is_actionable


# --- historical --------------------------------------------------------------------------

@pytest.mark.parametrize("text,term", [
    ("كانت تشتكي من صداع قبل سنة", "صداع"),
    ("عندها نزيف في حملها السابق", "نزيف"),
    ("سبق أن صار عندها تسمم حمل", "تسمم حمل"),
])
def test_past_findings_are_historical(text, term):
    result = at(text, term)
    assert result.assertion == "historical"
    assert not result.is_actionable


# --- family -------------------------------------------------------------------------------

@pytest.mark.parametrize("text,term", [
    ("أمها عندها سكري", "سكري"),
    ("التاريخ العائلي يبين ارتفاع ضغط", "ارتفاع ضغط"),
    ("أختها صار عندها تسمم حمل", "تسمم حمل"),
])
def test_relatives_findings_are_family(text, term):
    result = at(text, term)
    assert result.assertion == "family"
    assert not result.is_actionable


def test_family_wins_over_negation():
    """Whose body it is about is decided before whether it was denied."""
    assert at("أمها ما عندها سكري", "سكري").assertion == "family"


# --- planned ----------------------------------------------------------------------------

@pytest.mark.parametrize("text,term", [
    ("رح نعطيها سلفات المغنيسيوم", "سلفات المغنيسيوم"),
    ("سنطلب تحليل بول وزراعة", "تحليل بول"),
    ("ننصح بمتابعة الضغط يومياً", "الضغط"),
])
def test_intended_actions_are_planned(text, term):
    result = at(text, term)
    assert result.assertion == "planned"
    assert not result.is_actionable


def test_a_measurement_is_present_even_when_a_plan_follows():
    text = "ضغط الدم 150 على 90"
    assert at(text, "ضغط الدم").assertion == "present"


# --- precedence -----------------------------------------------------------------------

def test_hypothetical_beats_negation():
    """'إذا ما تحسن الصداع' has not happened; calling it absent would be wrong too."""
    assert at("إذا ما تحسن الصداع راجعينا", "الصداع").assertion == "hypothetical"


def test_backwards_compatible_helper_still_works():
    found, asserted = find_negation_aware("المريضة حامل بالأسبوع 30", ("حامل",))
    assert found and asserted

    found, asserted = find_negation_aware("المريضة ليست حامل", ("حامل",))
    assert found and not asserted

    found, asserted = find_negation_aware("إذا صرتِ حامل راجعينا", ("حامل",))
    assert found and not asserted


def test_one_actionable_mention_survives_another_being_hypothetical():
    text = "عندها صداع الآن، وإذا صار صداع أشد راجعينا"
    found, asserted = find_negation_aware(text, ("صداع",))
    assert found and asserted


def test_is_negated_remains_available_for_callers_that_only_need_it():
    assert is_negated("لا يوجد نزيف", 8)
    assert not is_negated("يوجد نزيف", 5)
