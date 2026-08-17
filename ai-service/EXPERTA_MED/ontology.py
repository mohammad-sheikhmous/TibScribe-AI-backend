"""Ontology: the closed vocabulary of clinical entity codes the KBS reasons over.

Every code has an Arabic display name and a category. Rules and the knowledge
graph only ever reference these codes — never raw text — so the reasoning layer
stays auditable. Curated for the OB/GYN domain of the 21-label classifier; needs
clinician sign-off before clinical use (same status as SOAP_MAPPING).
"""
from __future__ import annotations

# code -> {"ar": Arabic name, "category": symptom|vital|lab|test|condition|drug|action}
ENTITIES: dict[str, dict[str, str]] = {
    # --- symptoms ---
    "headache": {"ar": "صداع", "category": "symptom"},
    "blurred_vision": {"ar": "تشوش بالرؤية", "category": "symptom"},
    "epigastric_pain": {"ar": "ألم فم المعدة", "category": "symptom"},
    "edema": {"ar": "تورم/وذمة", "category": "symptom"},
    "dizziness": {"ar": "دوخة", "category": "symptom"},
    "fatigue": {"ar": "إرهاق عام", "category": "symptom"},
    "reduced_fetal_movement": {"ar": "قلة حركة الجنين", "category": "symptom"},
    "vaginal_bleeding": {"ar": "نزيف مهبلي", "category": "symptom"},
    "contractions": {"ar": "تقلصات/طلق", "category": "symptom"},
    "water_breaking": {"ar": "نزول ماء الأغشية", "category": "symptom"},
    "fever": {"ar": "حرارة/حمى", "category": "symptom"},
    "dysuria": {"ar": "حرقة بالتبول", "category": "symptom"},
    "itching": {"ar": "حكة", "category": "symptom"},
    "shortness_of_breath": {"ar": "ضيق نفس", "category": "symptom"},
    "palpitations": {"ar": "خفقان", "category": "symptom"},
    "abdominal_pain": {"ar": "ألم بالبطن", "category": "symptom"},
    # --- vitals / measurements ---
    "bp": {"ar": "ضغط الدم", "category": "vital"},
    "bp_high": {"ar": "ارتفاع ضغط الدم", "category": "finding"},
    "temp": {"ar": "الحرارة", "category": "vital"},
    "temp_high": {"ar": "ارتفاع الحرارة", "category": "finding"},
    # --- labs ---
    "urine_protein": {"ar": "زلال/بروتين البول", "category": "lab"},
    "hemoglobin": {"ar": "الهيموغلوبين", "category": "lab"},
    "hba1c": {"ar": "السكر التراكمي", "category": "lab"},
    # --- conditions ---
    "preeclampsia": {"ar": "تسمم الحمل", "category": "condition"},
    "gdm": {"ar": "سكري الحمل", "category": "condition"},
    "anemia": {"ar": "فقر الدم", "category": "condition"},
    "infection": {"ar": "عدوى", "category": "condition"},
    "uti": {"ar": "التهاب المسالك البولية", "category": "condition"},
    "hypertension": {"ar": "ارتفاع ضغط الدم المزمن", "category": "condition"},
    "photophobia": {"ar": "حساسية من الضوء (رهاب الضوء)", "category": "symptom"},
    "preterm_labor": {"ar": "ولادة مبكرة/طلق مبكر", "category": "condition"},
    "postpartum_hemorrhage": {"ar": "نزف ما بعد الولادة", "category": "condition"},
    "puerperal_infection": {"ar": "إنتان النفاس", "category": "condition"},
    # --- tests (orderable) ---
    "bp_measurement": {"ar": "قياس ضغط الدم", "category": "test"},
    "urine_protein_test": {"ar": "تحليل زلال البول", "category": "test"},
    "cbc": {"ar": "تعداد الدم الكامل CBC", "category": "test"},
    "liver_function": {"ar": "وظائف الكبد", "category": "test"},
    "fasting_glucose": {"ar": "سكر صائم", "category": "test"},
    "hba1c_test": {"ar": "تحليل السكر التراكمي", "category": "test"},
    "ctg": {"ar": "تخطيط قلب الجنين CTG", "category": "test"},
    "obstetric_ultrasound": {"ar": "تصوير (سونار) توليدي", "category": "test"},
    "blood_culture": {"ar": "زرع دم", "category": "test"},
    "urine_culture": {"ar": "زرع بول", "category": "test"},
    "wound_exam": {"ar": "فحص الجرح/مكان العملية", "category": "test"},
}

# drug code -> family (used for allergy cross-reactivity joins in rules)
DRUG_FAMILY: dict[str, str] = {
    "penicillin": "penicillin_family",
    "amoxicillin": "penicillin_family",
    "ampicillin": "penicillin_family",
    "levofloxacin": "fluoroquinolone",
    "ciprofloxacin": "fluoroquinolone",
    "ibuprofen": "nsaid",
    "diclofenac": "nsaid",
    "naproxen": "nsaid",
    "amlodipine": "ccb",
    "iron": "iron",
    "folic_acid": "folic_acid",
    "magnesium_sulfate": "magnesium_sulfate",
    "metformin": "metformin",
    "insulin": "insulin",
    "aspirin": "aspirin",
}

DRUG_AR: dict[str, str] = {
    "penicillin": "بنسلين",
    "amoxicillin": "أموكسيسيلين",
    "iron": "حديد",
    "folic_acid": "حمض الفوليك",
    "magnesium_sulfate": "سلفات المغنيسيوم",
    "metformin": "ميتفورمين",
    "insulin": "أنسولين",
    "aspirin": "أسبرين",
    "levofloxacin": "ليفوفلوكساسين",
    "ibuprofen": "إيبوبروفين (بروفين)",
    "diclofenac": "ديكلوفيناك",
    "amlodipine": "أملوديبين",
}


def ar(code: str) -> str:
    """Arabic display name for any entity/drug code (falls back to the code)."""
    if code in ENTITIES:
        return ENTITIES[code]["ar"]
    return DRUG_AR.get(code, code)


def drug_family(drug_code: str) -> str:
    """Family for cross-reactivity matching; unknown drugs are their own family."""
    return DRUG_FAMILY.get(drug_code, drug_code)
