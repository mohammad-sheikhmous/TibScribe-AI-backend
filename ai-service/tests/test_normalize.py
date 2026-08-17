import pytest

from app.core.nlp.normalize import get_preprocessing_mode, normalize_arabic

# Some assertions pin the exact output of the manual fallback normalizer (alef/taa/
# diacritic folding). When the official ArabertPreprocessor is installed the standard
# normalization differs by design, so those are skipped — the abbreviation-layer tests
# below run in BOTH modes (abbreviation expansion happens before either backend).
MANUAL = get_preprocessing_mode() == "manual"
manual_only = pytest.mark.skipif(not MANUAL, reason="manual-fallback normalization only")


@manual_only
def test_alef_variants_unified():
    assert normalize_arabic("آأإ") == "ااا"


@manual_only
def test_taa_marbuta_and_alef_maqsura():
    assert normalize_arabic("مدرسة") == "مدرسه"
    assert normalize_arabic("مستشفى") == "مستشفي"


@manual_only
def test_diacritics_stripped():
    assert normalize_arabic("مَرِيضٌ") == "مريض"


@manual_only
def test_plain_text_unchanged():
    assert normalize_arabic("صداع شديد") == "صداع شديد"


# --- Latin medical-abbreviation handling (mode-independent: runs before either backend) ---

def test_standalone_abbreviation_expanded():
    assert "تسمم الحمل" in normalize_arabic("Preeclampsia شديدة")


def test_multichar_abbreviation_expanded():
    assert "التصوير بالرنين المغناطيسي" in normalize_arabic("عملنا MRI للدماغ")


def test_abbreviation_not_matched_inside_longer_latin_word():
    # "A"/"G"/"P" must NOT corrupt these; unknown drug/disease names pass through intact.
    assert "Anaphylaxis" in normalize_arabic("طوارئ Anaphylaxis")
    assert "Atorvastatin" in normalize_arabic("صرفنا Atorvastatin حبة")
    assert "IgG" in normalize_arabic("Toxoplasma IgG عالي")
