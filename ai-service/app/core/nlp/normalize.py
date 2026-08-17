"""Arabic text normalization.

Single source of truth — previously duplicated verbatim in train_arabert.py
and inference.py. Both now import from here so training and inference can never
silently diverge on normalization.
"""
from __future__ import annotations

import re

# Common Latin medical abbreviations -> Arabic equivalents.
# Applied BEFORE Arabic normalization so the model sees consistent Arabic text.
_LATIN_ABBR: dict[str, str] = {
    "ECG": "اي سي جي",
    "EKG": "اي سي جي",
    "CBC": "سي بي سي",
    "TSH": "تي اس اتش",
    "FT4": "اف تي 4",
    "HbA1c": "هيموجلوبين السكر",
    "Hb": "هيموجلوبين",
    "BP": "ضغط الدم",
    "HR": "معدل ضربات القلب",
    "RR": "معدل التنفس",
    "SpO2": "أكسجة الدم",
    "BMI": "مؤشر كتلة الجسم",
    "BS": "سكر الدم",
    "US": "التصوير بالموجات فوق الصوتية",
    "U/S": "التصوير بالموجات فوق الصوتية",
    "CT": "التصوير المقطعي",
    "MRI": "التصوير بالرنين المغناطيسي",
    "IVF": "الحقن المجهري",
    "IUD": "اللولب الرحمي",
    "LMP": "أول يوم من آخر دورة شهرية",
    "EDD": "التاريخ المتوقع للولادة",
    "GA": "العمر الحملي",
    # NOTE: single-letter obstetric shorthand G / P / A (gravida / para / abortus)
    # was removed — blindly expanding a bare "A"/"G"/"P" corrupts unrelated Latin
    # words (Anaphylaxis, Atorvastatin, IgG, ...). Expand the combined GPA pattern
    # (e.g. "G3P2A1") separately if that signal is ever needed.
    "C-section": "ولادة قيصرية",
    "VBAC": "ولادة مهبلية بعد ولادة قيصرية",
    "PPD": "الاكتئاب ما بعد الولادة",
    "GDM": "السكري أثناء الحمل",
    "PIH": "ارتفاع ضغط الدم أثناء الحمل",
    "Preeclampsia": "تسمم الحمل",
    "DVT": "التهاب الأوردة العميقة",
    "PE": "انسداد الشريان الرئوي",
    "UTI": "التهاب المسالك البولية",
    "PID": "التهاب الحوض",
    "STI": "العدوى المنقولة جنسيا",
    "HPV": "فيروس الورم الحليمي البشري",
    "HIV": "فيروس نقص المناعة البشرية",
    "TORCH": "العدوى التي تنتقل من الأم للجنين",
    "OB": "التوليد",
    "GYN": "أمراض النساء",
    "OB/GYN": "التوليد وأمراض النساء",
    "Rx": "وصفة طبية",
    "Dx": "التشخيص",
    "Tx": "العلاج",
    "Fx": "الكسر",
    "hx": "التاريخ المرضي",
    "PMH": "التاريخ المرضي السابق",
    "FHx": "التاريخ العائلي",
    "SHx": "التاريخ الاجتماعي",
    "N/V": "غثيان وقيء",
    "SOB": "ضيق التنفس",
    "LOC": "فقدان الوعي",
    "I&O": "الدخول والخروج",
    "NPO": "صائم عن الطعام والشراب",
    "PRN": "حسب الحاجة",
    "BID": "مرتين يوميا",
    "TID": "ثلاث مرات يوميا",
    "QID": "أربع مرات يوميا",
    "QD": "مرة يوميا",
    "HS": "قبل النوم",
    "AC": "قبل الأكل",
    "PC": "بعد الأكل",
}

# Build a single compiled regex for all abbreviations. Longest-first so multi-char
# keys win over their prefixes, and Latin-alphanumeric lookarounds so an abbreviation
# only matches as a STANDALONE token — never inside a longer Latin word (which was
# corrupting drug/disease names). Case-sensitive on purpose.
_ABBR_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])("
    + "|".join(re.escape(k) for k in sorted(_LATIN_ABBR, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)


def _replace_latin_abbreviations(text: str) -> str:
    """Replace Latin medical abbreviations with their Arabic equivalents."""
    return _ABBR_PATTERN.sub(lambda m: _LATIN_ABBR[m.group(0)], text)


# --- Standard Arabic normalization ------------------------------------------------
# AraBERT (arabertv02) was pretrained with its OWN preprocessing (`ArabertPreprocessor`).
# Feeding it text normalized differently degrades results, so we prefer the official
# preprocessor when the `arabert` package is installed, and fall back to a manual
# normalization otherwise. The active mode is recorded in model_config.json at train
# time and checked at inference so a train/serve mismatch can't pass silently.

_MODE = None          # "arabert" | "manual" — resolved lazily on first use
_ARABERT_PREP = None


def _resolve_preprocessor() -> None:
    global _MODE, _ARABERT_PREP
    if _MODE is not None:
        return
    try:
        from arabert.preprocess import ArabertPreprocessor

        _ARABERT_PREP = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")
        _MODE = "arabert"
    except Exception:
        _ARABERT_PREP = None
        _MODE = "manual"


def get_preprocessing_mode() -> str:
    """Return the active standard-normalization backend: 'arabert' or 'manual'."""
    _resolve_preprocessor()
    return _MODE


def _manual_normalize(text: str) -> str:
    text = text.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return (
        text.replace("ّ", "")
        .replace("َ", "")
        .replace("ُ", "")
        .replace("ِ", "")
        .replace("ْ", "")
        .replace("ً", "")
        .replace("ٌ", "")
        .replace("ٍ", "")
    )


def normalize_arabic(text: str) -> str:
    text = _replace_latin_abbreviations(str(text))
    _resolve_preprocessor()
    if _ARABERT_PREP is not None:
        return _ARABERT_PREP.preprocess(text)
    return _manual_normalize(text)
