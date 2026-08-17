"""Preconception-care rules converted from ``Knowledge_Base_EN.md`` and ``Knowledge_Base_AR.md``.

The supplied source originally contained 48 atomic rules covering nutrition,
vaccination, genetics, environmental risks, infertility, FGM, STI, violence,
mental health, psychoactive substances, and tobacco.

The HIV Management rules FOLLOW5-R28, FOLLOW5-R29, FOLLOW5-R30, FOLLOW5-R31,
and FOLLOW5-R32 have been removed completely by request.

Output identifiers use the ``FOLLOW5-R*`` namespace to avoid confusion with the
legacy engine rules R01-R14.

IMPORTANT:
This module intentionally preserves the IF/THEN content of the attached source
files for all retained rules. No clinical recommendation has been changed.
"""

from __future__ import annotations

from experta import AS, MATCH, OR, Rule, TEST

from ..facts import ClinicalFact, PatientContext, RuleState
from ..suggestions import Suggestion

FOLLOW5_RULES_VERSION = "1.2.0"

RETIRED_RULE_IDS = frozenset({"FOLLOW5-R28", "FOLLOW5-R29", "FOLLOW5-R30", "FOLLOW5-R31", "FOLLOW5-R32"})


SOURCE_NAME_EN = "Knowledge_Base_EN.md"
SOURCE_NAME_AR = "Knowledge_Base_AR.md"
JUSTIFICATIONS_NAME = "Rule_Justifications.md"
METADATA_NAME = "Rule_Metadata.md"

CRITICAL_SALIENCE = 120
HIGH_SALIENCE = 80
MEDIUM_SALIENCE = 60

RULE_META = {
    "R1": (
        "nutrition_folic_acid",
        "Medication",
        "High",
        "High",
        "الحديد وحمض الفوليك قبل الحمل",
        "أوصِ بمكملات الحديد وحمض الفوليك للمرأة التي تخطط للحمل أو هي في سن الإنجاب، عبر التدعيم الغذائي أو الأقراص "
        "أو مساحيق المغذيات الدقيقة المحتوية على حمض الفوليك، مع تقديم المعلومات والتثقيف."
    ),
    "R2": (
        "nutrition_iron_deficiency",
        "Screening",
        "High",
        "High",
        "تحري فقر الدم بعوز الحديد",
        "تحرَّ عن فقر الدم عند وجود خطر لعوز الحديد أو فقر دم بعوز الحديد، وأوصِ بالحديد وحمض الفوليك والتدعيم الغذائي "
        "أو مساحيق المغذيات الدقيقة المحتوية على الحديد، مع التثقيف."
    ),
    "R3": (
        "nutrition_underweight",
        "Counseling",
        "High",
        "High",
        "تدبير نقص الوزن قبل الحمل",
        "عند BMI أقل من 18.5 مع التخطيط للحمل: قدم تثقيفاً غذائياً حول المخاطر، ومراقبة غذائية، وأغذية تكميلية كثيفة "
        "الطاقة والمغذيات."
    ),
    "R4": (
        "nutrition_overweight_obesity",
        "Counseling",
        "High",
        "High",
        "تدبير زيادة الوزن والسمنة قبل الحمل",
        "عند BMI يساوي أو يتجاوز 25 مع التخطيط للحمل: قدم تثقيفاً ومراقبة غذائية، وخفض المدخول الحراري، وزيادة النشاط "
        "البدني، وبرنامجاً منظماً لإنقاص الوزن، مع استمرار الرضاعة عند انطباقها."
    ),
    "R5": (
        "nutrition_diabetes",
        "Screening",
        "Critical",
        "High",
        "ضبط السكري قبل الحمل",
        "عند السكري من النوع الثاني أو تاريخ السكري الحملي مع التخطيط للحمل: قدم التثقيف، وافحص سكر الدم، واضبط السكري "
        "قبل الحمل وأثناءه وبعده، وأوصِ بالنشاط البدني والاستشارة الغذائية."
    ),
    "R6": (
        "nutrition_calcium",
        "Medication",
        "High",
        "High",
        "الكالسيوم عند خطر النقص قبل الحمل",
        "عند خطر نقص الكالسيوم مع التخطيط للحمل، أوصِ بمكملات الكالسيوم لتقليل خطر اضطرابات ارتفاع الضغط أثناء الحمل."
    ),
    "R7": (
        "vaccination_rubella",
        "Prevention",
        "Critical",
        "High",
        "لقاح الحصبة الألمانية قبل الحمل",
        "إذا لم تكن المرأة في سن الإنجاب مطعمة ضد الحصبة الألمانية وتخطط للحمل، أعطِ لقاحاً محتوياً على الحصبة "
        "الألمانية مثل R أو MR أو MMR."
    ),
    "R8": (
        "vaccination_rubella",
        "Counseling",
        "Critical",
        "High",
        "تأخير الحمل بعد لقاح الحصبة الألمانية",
        "بعد إعطاء لقاح محتوٍ على الحصبة الألمانية لامرأة تخطط للحمل، انصح بتجنب الحمل لمدة شهر واحد على الأقل."
    ),
    "R9": (
        "vaccination_tetanus",
        "Prevention",
        "High",
        "High",
        "استكمال لقاح الكزاز والدفتيريا قبل الحمل",
        "إذا لم يكتمل تطعيم الكزاز في الطفولة أو الأحمال السابقة مع التخطيط للحمل، أعطِ لقاح Td المحتوي على الكزاز "
        "والدفتيريا."
    ),
    "R10": (
        "vaccination_hepatitis_b",
        "Prevention",
        "High",
        "High",
        "استكمال لقاح التهاب الكبد B قبل الحمل",
        "إذا لم يكتمل لقاح التهاب الكبد B مع التخطيط للحمل، أعطِ اللقاح."
    ),
    "R11": (
        "genetic_counseling",
        "Assessment",
        "High",
        "High",
        "أخذ تاريخ عائلي ووراثي قبل الحمل",
        "عند تخطيط المرأة أو الزوجين للحمل، خذ تاريخاً عائلياً شاملاً يتضمن عمر الوالدين والأصل والقرابة والتاريخ "
        "الطبي والتوليدي السابق."
    ),
    "R12": (
        "genetic_counseling",
        "Diagnosis",
        "High",
        "High",
        "الإحالة والاستشارة الوراثية",
        "عند وجود تاريخ عائلي لمرض وراثي أو زواج أقارب أو نتائج حمل سلبية سابقة، ثبّت التشخيص الدقيق، وناقش خيارات "
        "تنظيم الأسرة، وقدم الاستشارة الوراثية وفحوص الحَملة والاختبارات."
    ),
    "R13": (
        "environment_radiation",
        "Counseling",
        "High",
        "High",
        "الحماية من الإشعاع قبل الحمل",
        "عند التعرض المهني أو البيئي أو الطبي للإشعاع المؤين مع التخطيط للحمل، قدم إرشاداً استباقياً واحمِ المرأة من "
        "التعرض غير الضروري."
    ),
    "R14": (
        "environment_pesticides",
        "Counseling",
        "High",
        "High",
        "تقليل التعرض للمبيدات قبل الحمل",
        "عند التعرض للمبيدات مع التخطيط للحمل، قدم إرشاداً استباقياً، وأوصِ بتجنب الاستخدام غير الضروري واقتراح "
        "بدائل."
    ),
    "R15": (
        "environment_metals",
        "Counseling",
        "High",
        "High",
        "الحماية من الرصاص والزئبق قبل الحمل",
        "عند التعرض للرصاص أو الزئبق مع التخطيط للحمل، قدم إرشاداً استباقياً وحماية من الرصاص في العمل، وإرشادات "
        "محلية للأسماك بحسب مستويات ميثيل الزئبق."
    ),
    "R16": (
        "environment_indoor_air",
        "Counseling",
        "High",
        "High",
        "تقليل تلوث الهواء المنزلي قبل الحمل",
        "عند التعرض لتلوث الهواء الداخلي من الوقود الصلب مع التخطيط للحمل، أوصِ بالمواقد المحسنة والوقود السائل أو "
        "الغازي الأنظف."
    ),
    "R17": (
        "infertility_management",
        "Screening",
        "High",
        "High",
        "تقييم تأخر الحمل بعد 6–12 شهراً",
        "إذا حاول الزوجان الحمل مدة 6 إلى 12 شهراً دون نجاح، ابدآ التحري والتشخيص بخوارزمية ذات تدخل محدود على مستوى "
        "الرعاية الأولية."
    ),
    "R18": (
        "infertility_management",
        "Referral",
        "High",
        "High",
        "إحالة تأخر الحمل بعد أكثر من 12 شهراً",
        "إذا تجاوزت محاولة الحمل 12 شهراً دون نجاح، أحِل للتقييم والتشخيص والتدبير في المستوى الثالثي."
    ),
    "R19": (
        "infertility_management",
        "Counseling",
        "Medium",
        "High",
        "تصحيح الاعتقاد بأن موانع الحمل تسبب العقم",
        "عند وجود اعتقاد خاطئ بأن موانع الحمل تسبب العقم، قدم إرشاداً حول الخصوبة والعقم، بما فيه أن نقص الخصوبة "
        "أثناء استعمال بعض الوسائل وبعد إيقافها قد يكون مؤقتاً."
    ),
    "R20": (
        "infertility_management",
        "Counseling",
        "Medium",
        "High",
        "التثقيف حول أسباب العقم",
        "عند سوء فهم الأسباب البيولوجية للعقم أو نقص الخصوبة، وضّح الأسباب القابلة وغير القابلة للوقاية وقدم الدعم "
        "للمشخَّصين بأسباب غير قابلة للوقاية."
    ),
    "R21": (
        "fgm",
        "Screening",
        "High",
        "High",
        "تحري خطر أو تاريخ ختان الإناث",
        "عند خطر ختان الإناث أو وجود تاريخ له، ناقش إيقاف الممارسة، وافحص المضاعفات الجنسية والخصوبية والولادية، وشجع "
        "منع انتقالها للجيل التالي وعالج العدوى أو أسباب العقم."
    ),
    "R22": (
        "fgm",
        "Counseling",
        "High",
        "High",
        "التثقيف حول مضاعفات ختان الإناث",
        "عند وجود ختان إناث، اشرح للمرأة والزوجين المضاعفات وطرق الوصول إلى العلاج."
    ),
    "R23": (
        "fgm",
        "Treatment",
        "Critical",
        "High",
        "فك الالتحام في الختان المغلق",
        "عند وجود ختان التحامي مغلق مع التخطيط للحمل أو في بداية الحمل، أجرِ فك الالتحام لتقليل مخاطر الولادة، وأزل "
        "الأكياس وعالج المضاعفات."
    ),
    "R24": (
        "pregnancy_spacing",
        "Counseling",
        "High",
        "High",
        "المباعدة بين الولادات",
        "عند رغبة المرأة أو الزوجين بتجنب فاصل ولادات أقل من 24 شهراً، اشرح مخاطر الفواصل القصيرة على الأم والطفل "
        "وقدم وسيلة منع حمل."
    ),
    "R25": (
        "sti",
        "Screening",
        "High",
        "High",
        "تحري العدوى المنقولة جنسياً قبل الحمل",
        "عند خطر الإصابة بعدوى منقولة جنسياً أو احتمال عدوى لا عرضية مع التخطيط للحمل، أجرِ تحرياً للعدوى اللاعرضية."
    ),
    "R26": (
        "sti",
        "Counseling",
        "High",
        "High",
        "الجنس الآمن والواقي قبل الحمل",
        "عند خطر العدوى المنقولة جنسياً مع التخطيط للحمل، شجع الممارسات الجنسية الأكثر أماناً واستعمال الواقي للحماية "
        "المزدوجة."
    ),
    "R27": (
        "sti",
        "Treatment",
        "Critical",
        "High",
        "علاج العدوى المنقولة جنسياً قبل الحمل",
        "عند تشخيص عدوى منقولة جنسياً مع التخطيط للحمل، قدم علاجاً متاحاً وميسوراً وخدمات صحية مرتبطة بالحالة."
    ),
    "R33": (
        "interpersonal_violence",
        "Assessment",
        "Critical",
        "High",
        "التعرف على علامات العنف قبل الحمل",
        "عند حضور المرأة لرعاية ما قبل الحمل، تعرّف على علامات عنف الشريك والعنف الجنسي."
    ),
    "R34": (
        "interpersonal_violence",
        "Referral",
        "Critical",
        "High",
        "رعاية الناجية من العنف",
        "عند كون المرأة ناجية من عنف الشريك أو العنف الجنسي، قدم الرعاية الطبية والنفسية الاجتماعية والدعم والإحالة "
        "المناسبة، بما في ذلك رعاية ما بعد الاغتصاب."
    ),
    "R35": (
        "mental_health",
        "Counseling",
        "Critical",
        "High",
        "تخطيط الحمل مع اضطراب نفسي أو صرع",
        "عند الاكتئاب أو الذهان أو الاضطراب ثنائي القطب أو الصرع مع التخطيط للحمل، ناقش تعديلات التدبير، وخطر "
        "الانتكاس، وتأثير المرض والأدوية على الأم والطفل."
    ),
    "R36": (
        "mental_health_depression",
        "Assessment",
        "High",
        "High",
        "تقييم الاكتئاب وخطر الانتحار قبل الحمل",
        "عند الاكتئاب مع التخطيط للحمل، قيّم المشكلات النفسية الاجتماعية والظروف المعيشية والمالية وأحداث الحياة "
        "الضاغطة وخطر الانتحار."
    ),
    "R37": (
        "mental_health_depression",
        "Treatment",
        "High",
        "Medium",
        "علاج الاكتئاب قبل الحمل",
        "عند الاكتئاب مع التخطيط للحمل والحاجة للعلاج، انظر في العلاج بين الشخصي أو التنشيط السلوكي أو العلاج المعرفي "
        "السلوكي؛ وإذا لزم الدواء فانظر في مضادات الاكتئاب ثلاثية الحلقات أو الفلوكسيتين واستشر اختصاصياً إن أمكن."
    ),
    "R38": (
        "mental_health_psychosis",
        "Medication",
        "Critical",
        "High",
        "تدبير الذهان قبل الحمل",
        "عند الذهان مع التخطيط للحمل، تجنب الاستخدام الروتيني لمضادات الذهان طويلة المفعول، وعالج بجرعات منخفضة فموية "
        "من هالوبيريدول أو كلوربرومازين، واشرح المخاطر واستشر اختصاصياً."
    ),
    "R39": (
        "mental_health_bipolar",
        "Medication",
        "Critical",
        "Medium",
        "تدبير الاضطراب ثنائي القطب قبل الحمل",
        "عند الاضطراب ثنائي القطب مع التخطيط للحمل، تجنب بدء مثبت مزاج، وانظر بحذر في هالوبيريدول بجرعة منخفضة واستشر "
        "اختصاصياً."
    ),
    "R40": (
        "mental_health_epilepsy",
        "Medication",
        "Critical",
        "High",
        "تجنب حمض الفالبرويك",
        "عند الصرع لدى امرأة في سن الإنجاب أو تخطط للحمل، تجنب حمض الفالبرويك."
    ),
    "R41": (
        "mental_health_epilepsy",
        "Medication",
        "Critical",
        "High",
        "حمض الفوليك بجرعة 5 ملغ مع أدوية الصرع",
        "عند الصرع مع تناول أدوية مضادة للصرع، انصح بحمض الفوليك 5 ملغ يومياً."
    ),
    "R42": (
        "mental_health_epilepsy",
        "Medication",
        "Critical",
        "High",
        "تجنب تعدد أدوية الصرع قبل الحمل",
        "عند الصرع مع التخطيط للحمل، تجنب تعدد الأدوية."
    ),
    "R43": (
        "psychoactive_substance_use",
        "Screening",
        "High",
        "High",
        "تحري تعاطي المواد قبل الحمل",
        "عند التخطيط للحمل، تحرَّ عن تعاطي المواد باستخدام أدوات مثل WHO ASSIST أو AUDIT."
    ),
    "R44": (
        "psychoactive_substance_use",
        "Referral",
        "High",
        "High",
        "التدخل والإحالة لتعاطي الكحول أو المخدرات",
        "عند استعمال ضار أو خطر للكحول أو المخدرات مع التخطيط للحمل، قدم تدخلاً موجزاً وإحالة للعلاج ومساعدة في تنظيم "
        "الأسرة."
    ),
    "R45": (
        "tobacco_use",
        "Screening",
        "High",
        "High",
        "تحري استعمال التبغ",
        "تحرَّ عن التدخين والتبغ غير المدخن في كل زيارة لدى من تخطط للحمل أو هي في سن الإنجاب باستخدام خطوات 5As."
    ),
    "R46": (
        "tobacco_use",
        "Treatment",
        "High",
        "High",
        "دعم الإقلاع عن التبغ",
        "عند استعمال التبغ، قدم نصيحة موجزة للإقلاع، واعرض العلاج الدوائي بما فيه بدائل النيكوتين عند توفرها، وأحِل "
        "للإرشاد السلوكي المكثف."
    ),
    "R47": (
        "tobacco_use",
        "Counseling",
        "High",
        "High",
        "الحماية من التدخين السلبي",
        "عند التعرض لدخان الآخرين، تحرَّ عن الضرر وقدم نصيحة حول مخاطر التدخين السلبي على المرأة الحامل والجنين."
    ),
    "R48": (
        "tobacco_use",
        "Referral",
        "High",
        "High",
        "مساعدة الشريك المدخن على الإقلاع",
        "إذا كان الزوج أو الشريك مدخناً، قدم له نصيحة موجزة وعلاجاً دوائياً متاحاً وإحالة للإرشاد السلوكي المكثف."
    ),
}


def _fact_evidence(fact) -> dict:
    get = fact.get if hasattr(fact, "get") else (lambda key, default=None: default)
    return {
        "item_id": get("item_id", ""),
        "text": get("text", ""),
        "report_index": get("report_index", None),
        "fact_code": get("code", ""),
        "value": get("value", None),
        "status": get("status", None),
    }


def _patient_context_pregnant_evidence(fact) -> dict:
    return {
        "item_id": "patient-context",
        "text": "PatientContext.pregnant=True",
        "report_index": None,
        "fact_code": "pregnant",
        "value": True,
        "status": None,
    }


class Follow5Rules:
    """Forty-three active preconception rules after removing FOLLOW5-R28..R32."""

    def _follow5_evidence(self, codes) -> list[dict]:
        wanted = set(codes)
        evidence = []

        for fact in getattr(self, "facts", {}).values():
            if isinstance(fact, ClinicalFact) and fact.get("code") in wanted:
                evidence.append(_fact_evidence(fact))

            elif isinstance(fact, PatientContext) and "pregnant" in wanted and fact.get("pregnant"):
                evidence.append(_patient_context_pregnant_evidence(fact))

        return evidence

    def _follow5_emit(self, rule_id: str, evidence_codes=()) -> None:
        category, rule_type, priority, confidence, title, detail = RULE_META[rule_id]
        full_id = f"FOLLOW5-{rule_id}"

        severity = {
            "Critical": "critical",
            "High": "high",
            "Medium": "medium",
        }.get(priority, "medium")

        self._emit(
            Suggestion(
                rule_id=full_id,
                rule_version=FOLLOW5_RULES_VERSION,
                severity=severity,
                condition=category,
                title_ar=title,
                detail_ar=detail,
                evidence=self._follow5_evidence(evidence_codes),
                chain=[
                    f"تحققت شروط القاعدة {full_id}",
                    f"الفئة: {category}؛ النوع: {rule_type}",
                    f"الأولوية: {priority}؛ قوة الدليل: {confidence}",
                    "إصدار توصية لما قبل الحمل للمراجعة السريرية",
                ],
                references=[
                    f"{SOURCE_NAME_EN} — {rule_id}",
                    f"{SOURCE_NAME_AR} — {rule_id}",
                    f"{JUSTIFICATIONS_NAME} — {rule_id}",
                    f"{METADATA_NAME} — {rule_id}",
                ],
            ),
            dedupe=(full_id, category),
        )
        self.declare(RuleState(code=full_id, category=category))

    # =====================================================================
    # 1. Nutrition
    # =====================================================================

    @Rule(
        OR(
            ClinicalFact(code="planning_pregnancy", value=True),
            ClinicalFact(code="reproductive_age", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r1(self):
        self._follow5_emit("R1", ("planning_pregnancy", "reproductive_age"))

    @Rule(
        OR(
            ClinicalFact(code="iron_deficiency_risk", value=True),
            ClinicalFact(code="iron_deficiency_anemia", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r2(self):
        self._follow5_emit("R2", ("iron_deficiency_risk", "iron_deficiency_anemia"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        AS.bmi << ClinicalFact(code="bmi", value=MATCH.value),
        TEST(lambda value: float(value) < 18.5),
        salience=HIGH_SALIENCE,
    )
    def r3(self, bmi, value):
        self._follow5_emit("R3", ("planning_pregnancy", "bmi"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        AS.bmi << ClinicalFact(code="bmi", value=MATCH.value),
        TEST(lambda value: float(value) >= 25),
        salience=HIGH_SALIENCE,
    )
    def r4(self, bmi, value):
        self._follow5_emit("R4", ("planning_pregnancy", "bmi"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        OR(
            ClinicalFact(code="diabetes_type2", value=True),
            ClinicalFact(code="history_gdm", value=True),
        ),
        salience=CRITICAL_SALIENCE,
    )
    def r5(self):
        self._follow5_emit("R5", ("planning_pregnancy", "diabetes_type2", "history_gdm"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="calcium_deficiency_risk", value=True),
        salience=HIGH_SALIENCE,
    )
    def r6(self):
        self._follow5_emit("R6", ("planning_pregnancy", "calcium_deficiency_risk"))

    # =====================================================================
    # 2. Vaccination
    # =====================================================================

    @Rule(
        ClinicalFact(code="reproductive_age", value=True),
        ClinicalFact(code="rubella_not_vaccinated", value=True),
        ClinicalFact(code="planning_pregnancy", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r7(self):
        self._follow5_emit(
            "R7",
            ("reproductive_age", "rubella_not_vaccinated", "planning_pregnancy"),
        )

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="rubella_vaccine_received", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r8(self):
        self._follow5_emit("R8", ("planning_pregnancy", "rubella_vaccine_received"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="tetanus_not_fully_immunized", value=True),
        salience=HIGH_SALIENCE,
    )
    def r9(self):
        self._follow5_emit("R9", ("planning_pregnancy", "tetanus_not_fully_immunized"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="hepatitis_b_not_fully_vaccinated", value=True),
        salience=HIGH_SALIENCE,
    )
    def r10(self):
        self._follow5_emit(
            "R10",
            ("planning_pregnancy", "hepatitis_b_not_fully_vaccinated"),
        )

    # =====================================================================
    # 3. Genetic Counseling
    # =====================================================================

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        salience=HIGH_SALIENCE,
    )
    def r11(self):
        self._follow5_emit("R11", ("planning_pregnancy",))

    @Rule(
        OR(
            ClinicalFact(code="family_history_genetic_disease", value=True),
            ClinicalFact(code="consanguinity", value=True),
            ClinicalFact(code="adverse_pregnancy_outcomes", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r12(self):
        self._follow5_emit(
            "R12",
            (
                "family_history_genetic_disease",
                "consanguinity",
                "adverse_pregnancy_outcomes",
            ),
        )

    # =====================================================================
    # 4. Environmental Risks
    # =====================================================================

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="ionizing_radiation_exposure", value=True),
        salience=HIGH_SALIENCE,
    )
    def r13(self):
        self._follow5_emit(
            "R13",
            ("planning_pregnancy", "ionizing_radiation_exposure"),
        )

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="pesticide_exposure", value=True),
        salience=HIGH_SALIENCE,
    )
    def r14(self):
        self._follow5_emit("R14", ("planning_pregnancy", "pesticide_exposure"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        OR(
            ClinicalFact(code="lead_exposure", value=True),
            ClinicalFact(code="mercury_exposure", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r15(self):
        self._follow5_emit("R15", ("planning_pregnancy", "lead_exposure", "mercury_exposure"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="indoor_air_pollution_exposure", value=True),
        salience=HIGH_SALIENCE,
    )
    def r16(self):
        self._follow5_emit(
            "R16",
            ("planning_pregnancy", "indoor_air_pollution_exposure"),
        )

    # =====================================================================
    # 5. Infertility Management
    # =====================================================================

    @Rule(
        AS.months << ClinicalFact(code="infertility_attempt_months", value=MATCH.value),
        TEST(lambda value: 6 <= float(value) <= 12),
        salience=HIGH_SALIENCE,
    )
    def r17(self, months, value):
        self._follow5_emit("R17", ("infertility_attempt_months",))

    @Rule(
        AS.months << ClinicalFact(code="infertility_attempt_months", value=MATCH.value),
        TEST(lambda value: float(value) > 12),
        salience=HIGH_SALIENCE,
    )
    def r18(self, months, value):
        self._follow5_emit("R18", ("infertility_attempt_months",))

    @Rule(
        ClinicalFact(code="contraception_infertility_misconception", value=True),
        salience=MEDIUM_SALIENCE,
    )
    def r19(self):
        self._follow5_emit("R19", ("contraception_infertility_misconception",))

    @Rule(
        ClinicalFact(code="biological_infertility_misconception", value=True),
        salience=MEDIUM_SALIENCE,
    )
    def r20(self):
        self._follow5_emit("R20", ("biological_infertility_misconception",))

    # =====================================================================
    # 6. Female Genital Mutilation (FGM)
    # =====================================================================

    @Rule(
        OR(
            ClinicalFact(code="fgm_risk", value=True),
            ClinicalFact(code="fgm_history", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r21(self):
        self._follow5_emit("R21", ("fgm_risk", "fgm_history"))

    @Rule(
        ClinicalFact(code="fgm_present", value=True),
        salience=HIGH_SALIENCE,
    )
    def r22(self):
        self._follow5_emit("R22", ("fgm_present",))

    @Rule(
        ClinicalFact(code="infibulated_fgm", value=True),
        ClinicalFact(code="planning_pregnancy", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r23_planning(self):
        self._follow5_emit("R23", ("infibulated_fgm", "planning_pregnancy"))

    @Rule(
        ClinicalFact(code="infibulated_fgm", value=True),
        PatientContext(pregnant=True),
        ClinicalFact(code="early_pregnancy", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r23_early_pregnancy(self):
        self._follow5_emit(
            "R23",
            ("infibulated_fgm", "early_pregnancy", "pregnant"),
        )

    # =====================================================================
    # 7. Pregnancy Spacing
    # =====================================================================

    @Rule(
        ClinicalFact(code="desires_avoid_short_birth_interval", value=True),
        salience=HIGH_SALIENCE,
    )
    def r24(self):
        self._follow5_emit("R24", ("desires_avoid_short_birth_interval",))

    # =====================================================================
    # 8. Sexually Transmitted Infections (STI)
    # =====================================================================

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        OR(
            ClinicalFact(code="sti_risk", value=True),
            ClinicalFact(code="symptomless_sti", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r25(self):
        self._follow5_emit("R25", ("planning_pregnancy", "sti_risk", "symptomless_sti"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="sti_risk", value=True),
        salience=HIGH_SALIENCE,
    )
    def r26(self):
        self._follow5_emit("R26", ("planning_pregnancy", "sti_risk"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="sti_diagnosed", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r27(self):
        self._follow5_emit("R27", ("planning_pregnancy", "sti_diagnosed"))

    # =====================================================================
    # HIV Management rules FOLLOW5-R28..FOLLOW5-R32 removed completely.
    # =====================================================================

    # =====================================================================
    # 9. Interpersonal Violence
    # =====================================================================

    @Rule(
        ClinicalFact(code="preconception_care_visit", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r33(self):
        self._follow5_emit("R33", ("preconception_care_visit",))

    @Rule(
        OR(
            ClinicalFact(code="ipv_survivor", value=True),
            ClinicalFact(code="sexual_violence_survivor", value=True),
        ),
        salience=CRITICAL_SALIENCE,
    )
    def r34(self):
        self._follow5_emit("R34", ("ipv_survivor", "sexual_violence_survivor"))

    # =====================================================================
    # 10. Mental Health
    # =====================================================================

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        OR(
            ClinicalFact(code="depression", value=True),
            ClinicalFact(code="psychosis", value=True),
            ClinicalFact(code="bipolar_disorder", value=True),
            ClinicalFact(code="epilepsy", value=True),
        ),
        salience=CRITICAL_SALIENCE,
    )
    def r35(self):
        self._follow5_emit(
            "R35",
            (
                "planning_pregnancy",
                "depression",
                "psychosis",
                "bipolar_disorder",
                "epilepsy",
            ),
        )

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="depression", value=True),
        salience=HIGH_SALIENCE,
    )
    def r36(self):
        self._follow5_emit("R36", ("planning_pregnancy", "depression"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="depression", value=True),
        ClinicalFact(code="mental_health_treatment_required", value=True),
        salience=HIGH_SALIENCE,
    )
    def r37(self):
        self._follow5_emit(
            "R37",
            ("planning_pregnancy", "depression", "mental_health_treatment_required"),
        )

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="psychosis", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r38(self):
        self._follow5_emit("R38", ("planning_pregnancy", "psychosis"))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="bipolar_disorder", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r39(self):
        self._follow5_emit("R39", ("planning_pregnancy", "bipolar_disorder"))

    @Rule(
        ClinicalFact(code="epilepsy", value=True),
        OR(
            ClinicalFact(code="planning_pregnancy", value=True),
            ClinicalFact(code="reproductive_age", value=True),
        ),
        salience=CRITICAL_SALIENCE,
    )
    def r40(self):
        self._follow5_emit("R40", ("epilepsy", "planning_pregnancy", "reproductive_age"))

    @Rule(
        ClinicalFact(code="epilepsy", value=True),
        ClinicalFact(code="taking_antiepileptic_drugs", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r41(self):
        self._follow5_emit("R41", ("epilepsy", "taking_antiepileptic_drugs"))

    @Rule(
        ClinicalFact(code="epilepsy", value=True),
        ClinicalFact(code="planning_pregnancy", value=True),
        salience=CRITICAL_SALIENCE,
    )
    def r42(self):
        self._follow5_emit("R42", ("epilepsy", "planning_pregnancy"))

    # =====================================================================
    # 11. Psychoactive Substance Use
    # =====================================================================

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        salience=HIGH_SALIENCE,
    )
    def r43(self):
        self._follow5_emit("R43", ("planning_pregnancy",))

    @Rule(
        ClinicalFact(code="planning_pregnancy", value=True),
        ClinicalFact(code="hazardous_alcohol_or_drug_use", value=True),
        salience=HIGH_SALIENCE,
    )
    def r44(self):
        self._follow5_emit(
            "R44",
            ("planning_pregnancy", "hazardous_alcohol_or_drug_use"),
        )

    # =====================================================================
    # 12. Tobacco Use
    # =====================================================================

    @Rule(
        OR(
            ClinicalFact(code="planning_pregnancy", value=True),
            ClinicalFact(code="reproductive_age", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r45(self):
        self._follow5_emit("R45", ("planning_pregnancy", "reproductive_age"))

    @Rule(
        OR(
            ClinicalFact(code="tobacco_use", value=True),
            ClinicalFact(code="smokeless_tobacco_use", value=True),
        ),
        salience=HIGH_SALIENCE,
    )
    def r46(self):
        self._follow5_emit("R46", ("tobacco_use", "smokeless_tobacco_use"))

    @Rule(
        ClinicalFact(code="second_hand_smoke_exposure", value=True),
        salience=HIGH_SALIENCE,
    )
    def r47(self):
        self._follow5_emit("R47", ("second_hand_smoke_exposure",))

    @Rule(
        ClinicalFact(code="partner_smokes", value=True),
        salience=HIGH_SALIENCE,
    )
    def r48(self):
        self._follow5_emit("R48", ("partner_smokes",))