"""Label metadata contract: SOAP coverage and deprecated-label back-compatibility.

The taxonomy changed on 2026-07-26 (pregnancy_nutrition -> nutrition, 21 -> 20
labels). These tests pin the two properties that must survive any future change:
nothing active is unmapped, and nothing historical silently lands in the wrong
section.
"""
from app.core.nlp.sections import (
    ACTIVE_LABELS,
    DEPRECATED_LABELS,
    MEDICAL_SECTIONS,
    SOAP_MAPPING,
    SOAP_ORDER,
    label_ar,
    soap_for_label,
)


def test_active_label_set_is_twenty():
    assert len(ACTIVE_LABELS) == 20
    assert "pregnancy_nutrition" not in ACTIVE_LABELS


def test_every_active_label_has_a_soap_section_and_arabic_name():
    for label in ACTIVE_LABELS:
        assert SOAP_MAPPING.get(label) in SOAP_ORDER, f"{label} has no SOAP section"
        assert label_ar(label) != label, f"{label} has no Arabic display name"


def test_deprecated_label_still_maps_to_its_original_section():
    """A report produced by an older checkpoint must not drift into the wrong section."""
    for old in DEPRECATED_LABELS:
        assert old in SOAP_MAPPING, f"{old} was removed — old reports would fall back"
        assert soap_for_label(old) == SOAP_MAPPING[old]
    assert soap_for_label("pregnancy_nutrition") == "plan"  # not the 'assessment' fallback


def test_deprecated_labels_point_at_live_replacements():
    for old, new in DEPRECATED_LABELS.items():
        assert new in ACTIVE_LABELS, f"{old} redirects to {new}, which is not active"


def test_unknown_label_falls_back_to_assessment(caplog):
    assert soap_for_label("brand_new_label_from_a_retrain") == "assessment"
    assert any("no SOAP mapping" in r.message for r in caplog.records)


def test_mapping_has_no_labels_outside_the_display_table():
    assert set(SOAP_MAPPING) <= set(MEDICAL_SECTIONS)
