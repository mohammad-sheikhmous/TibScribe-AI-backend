"""Pregnancy follow-up rules converted from ``2_kb_rules.md``.

Source basis:
    WHO Recommendations on Antenatal Care for a Positive Pregnancy Experience
    (2016), as structured in the supplied Markdown file.

The rules preserve the conditions, priorities, confidence labels, severities, and
recommendations written in the supplied source. They are decision-support
suggestions for clinician review, not autonomous treatment orders.
"""
from __future__ import annotations

from experta import AS, MATCH, NOT, Rule, TEST

from ..facts import ClinicalFact, PatientContext, RuleState
from ..suggestions import Suggestion

FOLLOW2_RULES_VERSION = "1.0.0"
SOURCE_NAME = "2_kb_rules.md — WHO ANC 2016"


def _fact_evidence(fact) -> dict:
    """Return a JSON-safe evidence object from an Experta fact."""
    get = fact.get if hasattr(fact, "get") else (lambda key, default=None: default)
    return {
        "item_id": get("item_id", ""),
        "text": get("text", ""),
        "report_index": get("report_index", None),
        "fact_code": get("code", ""),
        "value": get("value", None),
        "status": get("status", None),
    }


class Follow2Rules:
    """Twenty-eight atomic rules N1–N28 from the second ANC file."""

    def _follow2_evidence(self, codes: set[str] | tuple[str, ...] | list[str]) -> list[dict]:
        wanted = set(codes)
        evidence: list[dict] = []
        for fact in getattr(self, "facts", {}).values():
            if isinstance(fact, ClinicalFact) and fact.get("code") in wanted:
                evidence.append(_fact_evidence(fact))
        return evidence

    def _follow2_emit(
        self,
        *,
        rule_id: str,
        severity: str,
        category: str,
        rule_type: str,
        priority: str,
        confidence: str,
        title: str,
        detail: str,
        source: str,
        evidence_codes: tuple[str, ...] = (),
        chain: list[str] | None = None,
        mark_state: bool = True,
    ) -> None:
        full_id = f"FOLLOW2-{rule_id}"
        evidence = self._follow2_evidence(evidence_codes)
        default_chain = [
            f"تحققت شروط القاعدة {full_id}",
            f"الفئة: {category}؛ النوع: {rule_type}",
            f"الأولوية: {priority}؛ قوة الدليل: {confidence}",
            "إصدار توصية للمراجعة السريرية",
        ]
        self._emit(
            Suggestion(
                rule_id=full_id,
                rule_version=FOLLOW2_RULES_VERSION,
                severity=severity,
                condition=category,
                title_ar=title,
                detail_ar=detail,
                evidence=evidence,
                chain=chain or default_chain,
                references=[f"{SOURCE_NAME} — {rule_id} — {source}"],
            ),
            dedupe=(full_id, category),
        )
        if mark_state:
            self.declare(RuleState(code=full_id, category=category))

    # ------------------------------------------------------------------
    # 1. Detailed nutrition — N1..N10
    # ------------------------------------------------------------------
    @Rule(PatientContext(pregnant=True), salience=30)
    def n1_healthy_diet_activity(self):
        self._follow2_emit(
            rule_id="N1", severity="low", category="nutrition",
            rule_type="Counseling", priority="Medium", confidence="High",
            title="الأكل الصحي والنشاط البدني أثناء الحمل",
            detail=(
                "أرشد حول الأكل الصحي والنشاط البدني في جميع التماسات الحمل، "
                "بهدف تجنب الزيادة المفرطة بالوزن."
            ),
            source="A.1.1",
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="undernourished_community", value=True),
        salience=70,
    )
    def n2_energy_protein_counseling(self):
        self._follow2_emit(
            rule_id="N2", severity="low", category="nutrition",
            rule_type="Counseling", priority="Medium", confidence="Medium",
            title="زيادة الطاقة والبروتين في مناطق سوء التغذية",
            detail=(
                "ثقّف المرأة حول زيادة مدخول الطاقة والبروتين اليومي لتقليل خطر "
                "انخفاض وزن المولود."
            ),
            source="A.1.2", evidence_codes=("undernourished_community",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="undernourished_community", value=True),
        salience=69,
    )
    def n3_balanced_energy_protein_supplements(self):
        self._follow2_emit(
            rule_id="N3", severity="low", category="nutrition",
            rule_type="Prevention", priority="Medium", confidence="Medium",
            title="مكملات طاقة وبروتين متوازنة",
            detail=(
                "أعطِ مكملات طاقة وبروتين متوازنة لتقليل خطر الإملاص وصغر حجم "
                "الجنين بالنسبة لعمر الحمل."
            ),
            source="A.1.3", evidence_codes=("undernourished_community",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="undernourished_community", value=True),
        salience=68,
    )
    def n4_avoid_high_protein_supplements(self):
        self._follow2_emit(
            rule_id="N4", severity="low", category="nutrition",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم التوصية بمكملات البروتين عالي التركيز",
            detail=(
                "لا تُوصِ بمكملات البروتين عالي التركيز؛ يذكر المصدر عدم ثبوت "
                "فائدتها مع احتمال الضرر."
            ),
            source="A.1.4", evidence_codes=("undernourished_community",),
        )

    @Rule(
        PatientContext(pregnant=True),
        NOT(RuleState(code="FOLLOW2-N6")),
        salience=29,
    )
    def n5_daily_iron_folic_acid(self):
        self._follow2_emit(
            rule_id="N5", severity="low", category="nutrition",
            rule_type="Medication", priority="High", confidence="High",
            title="الحديد وحمض الفوليك يومياً",
            detail=(
                "أعطِ حديداً عنصرياً 30–60 ملغ يومياً مع حمض فوليك 400 ميكروغرام "
                "يومياً، للوقاية من فقر دم الأم وإنتان النفاس وانخفاض وزن الولادة "
                "والولادة المبكرة."
            ),
            source="A.2.1",
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="daily_iron_not_tolerated", value=True),
        AS.prevalence << ClinicalFact(code="community_anemia_prevalence_percent", value=MATCH.value),
        TEST(lambda value: float(value) < 20),
        salience=75,
    )
    def n6_weekly_iron_folic_acid(self, prevalence, value):
        self._follow2_emit(
            rule_id="N6", severity="low", category="nutrition",
            rule_type="Medication", priority="Medium", confidence="Medium",
            title="بديل أسبوعي للحديد وحمض الفوليك",
            detail=(
                "عند عدم تحمل الحديد اليومي وانتشار فقر الدم بالمجتمع أقل من 20%، "
                "أعطِ حديداً 120 ملغ مع حمض فوليك 2800 ميكروغرام مرة واحدة أسبوعياً."
            ),
            source="A.2.2",
            evidence_codes=("daily_iron_not_tolerated", "community_anemia_prevalence_percent"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="low_dietary_calcium_intake", value=True),
        salience=72,
    )
    def n7_calcium_supplementation(self):
        self._follow2_emit(
            rule_id="N7", severity="low", category="nutrition",
            rule_type="Medication", priority="Medium", confidence="High",
            title="مكملات الكالسيوم عند انخفاض المدخول",
            detail=(
                "أعطِ كالسيوم عنصرياً فموياً بجرعة 1.5–2.0 غرام يومياً للوقاية "
                "من تسمم الحمل."
            ),
            source="A.3", evidence_codes=("low_dietary_calcium_intake",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="severe_vitamin_a_deficiency_area", value=True),
        salience=71,
    )
    def n8_vitamin_a_deficiency_area(self):
        self._follow2_emit(
            rule_id="N8", severity="low", category="nutrition",
            rule_type="Medication", priority="Medium", confidence="Medium",
            title="فيتامين A في مناطق النقص الشديد",
            detail="أعطِ فيتامين A للوقاية من العشى الليلي وفق سياق النقص الشديد بالمجتمع.",
            source="A.4", evidence_codes=("severe_vitamin_a_deficiency_area",),
        )

    @Rule(
        PatientContext(pregnant=True),
        AS.caffeine << ClinicalFact(code="daily_caffeine_mg", value=MATCH.value),
        TEST(lambda value: float(value) > 300),
        salience=73,
    )
    def n9_reduce_caffeine(self, caffeine, value):
        self._follow2_emit(
            rule_id="N9", severity="low", category="nutrition",
            rule_type="Counseling", priority="Medium", confidence="High",
            title="تقليل الكافيين أثناء الحمل",
            detail=(
                "أرشد المرأة إلى تقليل الكافيين عندما يتجاوز الاستهلاك اليومي "
                "300 ملغ، لتقليل خطر فقدان الحمل وانخفاض وزن الولادة."
            ),
            source="A.10", evidence_codes=("daily_caffeine_mg",),
        )

    @Rule(PatientContext(pregnant=True), salience=28)
    def n10_avoid_routine_micronutrients(self):
        self._follow2_emit(
            rule_id="N10", severity="low", category="nutrition",
            rule_type="Prevention", priority="Low", confidence="High",
            title="مكملات لا يُوصى بها روتينياً",
            detail=(
                "لا تُوصِ روتينياً بالزنك، أو المكملات متعددة المغذيات الدقيقة، "
                "أو فيتامين B6 وحده لتحسين نتائج الحمل، أو فيتامينات E أو C أو D."
            ),
            source="A.5–A.9",
        )

    # ------------------------------------------------------------------
    # 2. Gestational/overt diabetes — N11..N12
    # ------------------------------------------------------------------
    def _emit_n11(self, evidence_code: str) -> None:
        self._follow2_emit(
            rule_id="N11", severity="medium", category="gestational_diabetes",
            rule_type="Diagnosis", priority="Medium", confidence="High",
            title="تشخيص سكري الحمل",
            detail=(
                "شخّص سكرياً حملياً (GDM)، وسجّل التشخيص، وأحِل لمتابعة متخصصة "
                "بالسكري والحمل."
            ),
            source="B.1.4", evidence_codes=(evidence_code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        AS.result << ClinicalFact(code="fasting_glucose_mmol_l", value=MATCH.value),
        TEST(lambda value: 5.1 <= float(value) <= 6.9),
        NOT(RuleState(code="FOLLOW2-N12")),
        salience=110,
    )
    def n11_fasting_glucose(self, result, value):
        self._emit_n11("fasting_glucose_mmol_l")

    @Rule(
        PatientContext(pregnant=True),
        AS.result << ClinicalFact(code="ogtt_1h_glucose_mmol_l", value=MATCH.value),
        TEST(lambda value: float(value) > 10.0),
        NOT(RuleState(code="FOLLOW2-N12")),
        salience=110,
    )
    def n11_one_hour_glucose(self, result, value):
        self._emit_n11("ogtt_1h_glucose_mmol_l")

    @Rule(
        PatientContext(pregnant=True),
        AS.result << ClinicalFact(code="ogtt_2h_glucose_mmol_l", value=MATCH.value),
        TEST(lambda value: 8.5 <= float(value) <= 11.0),
        NOT(RuleState(code="FOLLOW2-N12")),
        salience=110,
    )
    def n11_two_hour_glucose(self, result, value):
        self._emit_n11("ogtt_2h_glucose_mmol_l")

    def _emit_n12(self, evidence_codes: tuple[str, ...]) -> None:
        self._follow2_emit(
            rule_id="N12", severity="high", category="overt_diabetes_in_pregnancy",
            rule_type="Diagnosis", priority="High", confidence="High",
            title="سكري ظاهر أثناء الحمل",
            detail=(
                "شخّص سكرياً ظاهراً أثناء الحمل بأولوية أعلى من N11، وأحِل عاجلاً "
                "للتقييم والمتابعة المتخصصة بالسكري."
            ),
            source="B.1.4", evidence_codes=evidence_codes,
        )

    @Rule(
        PatientContext(pregnant=True),
        AS.result << ClinicalFact(code="fasting_glucose_mmol_l", value=MATCH.value),
        TEST(lambda value: float(value) > 7.0),
        salience=125,
    )
    def n12_fasting_glucose(self, result, value):
        self._emit_n12(("fasting_glucose_mmol_l",))

    @Rule(
        PatientContext(pregnant=True),
        AS.result << ClinicalFact(code="ogtt_2h_glucose_mmol_l", value=MATCH.value),
        TEST(lambda value: float(value) > 11.1),
        salience=125,
    )
    def n12_two_hour_glucose(self, result, value):
        self._emit_n12(("ogtt_2h_glucose_mmol_l",))

    @Rule(
        PatientContext(pregnant=True),
        AS.result << ClinicalFact(code="random_glucose_mmol_l", value=MATCH.value),
        TEST(lambda value: float(value) > 11.1),
        ClinicalFact(code="overt_diabetes_symptoms", value=True),
        salience=125,
    )
    def n12_random_glucose_with_symptoms(self, result, value):
        self._emit_n12(("random_glucose_mmol_l", "overt_diabetes_symptoms"))

    # ------------------------------------------------------------------
    # 3. Asymptomatic bacteriuria — N13..N15
    # ------------------------------------------------------------------
    def _emit_n13(self, evidence_codes: tuple[str, ...]) -> None:
        self._follow2_emit(
            rule_id="N13", severity="medium", category="asymptomatic_bacteriuria",
            rule_type="Diagnosis", priority="Medium", confidence="Medium",
            title="تشخيص البكتيريا اللاعرضية في البول",
            detail=(
                "شخّص بكتيريا لاعرضية في البول (ASB). يذكر المصدر أن الفحص يتم "
                "روتينياً حتى في غياب الأعراض."
            ),
            source="B.1.2", evidence_codes=evidence_codes,
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="midstream_urine_culture_positive", value=True),
        salience=105,
    )
    def n13_positive_urine_culture(self):
        self._emit_n13(("midstream_urine_culture_positive",))

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="urine_culture_unavailable", value=True),
        ClinicalFact(code="midstream_urine_gram_stain_positive", value=True),
        salience=105,
    )
    def n13_positive_gram_stain(self):
        self._emit_n13(("urine_culture_unavailable", "midstream_urine_gram_stain_positive"))

    @Rule(RuleState(code="FOLLOW2-N13"), salience=100)
    def n14_asb_antibiotic(self):
        self._follow2_emit(
            rule_id="N14", severity="medium", category="asymptomatic_bacteriuria",
            rule_type="Medication", priority="High", confidence="High",
            title="علاج البكتيريا اللاعرضية لمدة سبعة أيام",
            detail=(
                "أعطِ مضاداً حيوياً فموياً لمدة سبعة أيام كاملة للوقاية من استمرار "
                "البكتيريا والولادة المبكرة وانخفاض وزن الولادة."
            ),
            source="C.1",
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="recurrent_uti_history", value=True),
        salience=65,
    )
    def n15_no_continuous_uti_prophylaxis(self):
        self._follow2_emit(
            rule_id="N15", severity="low", category="asymptomatic_bacteriuria",
            rule_type="Prevention", priority="Low", confidence="Low",
            title="عدم التوصية بالوقاية المستمرة من التهاب البول",
            detail=(
                "لا تُوصِ بمضاد حيوي وقائي مستمر لمنع تكرر التهابات المسالك؛ "
                "يقتصر ذلك وفق المصدر على سياق بحثي صارم."
            ),
            source="C.2", evidence_codes=("recurrent_uti_history",),
        )

    # ------------------------------------------------------------------
    # 4. Intimate-partner violence — N16
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="ipv_related_condition_suspected", value=True),
        ClinicalFact(code="ipv_response_system_available", value=True),
        salience=90,
    )
    def n16_ipv_clinical_enquiry(self):
        self._follow2_emit(
            rule_id="N16", severity="medium", category="intimate_partner_violence",
            rule_type="Screening", priority="Medium", confidence="Medium",
            title="استفسار سريري آمن عن عنف الشريك",
            detail=(
                "استفسر بلطف وخصوصية عن احتمال عنف الشريك فقط عند توفر الخصوصية، "
                "وتدريب الطاقم، واستجابة داعمة، ومسار إحالة فعلي."
            ),
            source="B.1.3",
            evidence_codes=("ipv_related_condition_suspected", "ipv_response_system_available"),
        )

    # ------------------------------------------------------------------
    # 5. Active TB screening — N17
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        AS.incidence << ClinicalFact(code="tb_incidence_per_100k", value=MATCH.value),
        TEST(lambda value: float(value) >= 100),
        salience=88,
    )
    def n17_systematic_tb_screening(self, incidence, value):
        self._follow2_emit(
            rule_id="N17", severity="medium", category="active_tb_screening",
            rule_type="Screening", priority="Medium", confidence="High",
            title="الفحص المنهجي للسل النشط",
            detail=(
                "عندما يبلغ انتشار السل 100 لكل 100,000 نسمة أو أكثر، أجرِ فحصاً "
                "منهجياً للسل النشط لكل الحوامل في التماس الأول."
            ),
            source="B.1.8", evidence_codes=("tb_incidence_per_100k",),
        )

    # ------------------------------------------------------------------
    # 6. Fetal assessment — N18..N19
    # ------------------------------------------------------------------
    @Rule(
        AS.context << PatientContext(pregnant=True, ga_weeks=MATCH.ga),
        TEST(lambda ga: 0 < int(ga) < 24),
        salience=78,
    )
    def n18_ultrasound_before_24_weeks(self, context, ga):
        self._follow2_emit(
            rule_id="N18", severity="low", category="fetal_assessment",
            rule_type="Assessment", priority="High", confidence="High",
            title="موجة فوق صوتية واحدة قبل الأسبوع 24",
            detail=(
                "أجرِ موجة فوق صوتية واحدة قبل الأسبوع 24 لتقدير عمر الحمل، وكشف "
                "تشوهات الجنين والحمل المتعدد، وتقليل حث الولادة غير الضروري."
            ),
            source="B.2.4",
            chain=[
                "PatientContext.pregnant = True",
                f"عمر الحمل = {int(ga)} أسبوع (<24)",
                "FOLLOW2-N18 → توصية بموجة فوق صوتية واحدة قبل الأسبوع 24",
                "الأولوية: High؛ قوة الدليل: High",
            ],
        )

    @Rule(PatientContext(pregnant=True), salience=27)
    def n19_avoid_routine_fetal_interventions(self):
        self._follow2_emit(
            rule_id="N19", severity="low", category="fetal_assessment",
            rule_type="Prevention", priority="Low", confidence="High",
            title="إجراءات جنينية لا تُوصى بها روتينياً",
            detail=(
                "لا تُوصِ روتينياً بعدّ حركة الجنين اليومي المنظم، أو استبدال جس "
                "البطن بقياس ارتفاع قاع الرحم، أو CTG قبل المخاض، أو دوبلر الأوعية "
                "الجنينية الروتيني."
            ),
            source="B.2.1–B.2.3, B.2.5",
        )

    # ------------------------------------------------------------------
    # 7. Anti-D immunoglobulin — N20
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="maternal_rh_negative", value=True),
        ClinicalFact(code="rh_unsensitized", value=True),
        salience=67,
    )
    def n20_anti_d_research_context(self):
        self._follow2_emit(
            rule_id="N20", severity="low", category="rh_prevention",
            rule_type="Prevention", priority="Low", confidence="Low",
            title="عدم التوصية الروتينية بمضاد Rh في الأسبوعين 28 و34",
            detail=(
                "للأم Rh سلبية وغير المحسّسة، لا تُوصِ روتينياً بإعطاء الغلوبولين "
                "المناعي المضاد لـRh في الأسبوعين 28 و34؛ يقتصر على سياق بحثي صارم "
                "أو حيث تتوفر الموارد وفق المصدر."
            ),
            source="C.3", evidence_codes=("maternal_rh_negative", "rh_unsensitized"),
        )

    # ------------------------------------------------------------------
    # 8. HIV PrEP — N21
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="hiv_negative", value=True),
        ClinicalFact(code="high_hiv_exposure_risk", value=True),
        salience=92,
    )
    def n21_hiv_prep(self):
        self._follow2_emit(
            rule_id="N21", severity="medium", category="hiv_prep",
            rule_type="Prevention", priority="Medium", confidence="Medium",
            title="عرض الوقاية قبل التعرض لـHIV",
            detail=(
                "اعرض العلاج الوقائي قبل التعرض الفموي (TDF) كخيار إضافي ضمن حزمة "
                "الوقاية المركبة في التماس الأول."
            ),
            source="C.7", evidence_codes=("hiv_negative", "high_hiv_exposure_risk"),
        )

    # ------------------------------------------------------------------
    # 9. Common physiological symptoms — N22..N27
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="early_pregnancy", value=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {"nausea", "vomiting"}),
        salience=76,
    )
    def n22_nausea_vomiting(self, trigger, code):
        self._follow2_emit(
            rule_id="N22", severity="low", category="physiological_symptoms",
            rule_type="Treatment", priority="Medium", confidence="High",
            title="الغثيان والقيء في بداية الحمل",
            detail=(
                "اعرض حسب تفضيل المرأة الزنجبيل و/أو البابونج و/أو فيتامين B6 و/أو "
                "الوخز بالإبر، وطمئنها بأن الأعراض عادة تزول في النصف الثاني من الحمل."
            ),
            source="D.1", evidence_codes=(code, "early_pregnancy"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="heartburn", value=True),
        salience=75,
    )
    def n23_heartburn(self):
        self._follow2_emit(
            rule_id="N23", severity="low", category="physiological_symptoms",
            rule_type="Treatment", priority="Medium", confidence="High",
            title="تدبير حرقة المعدة",
            detail=(
                "ابدأ بالإرشاد الغذائي ونمط الحياة؛ وإذا لم تستجب الأعراض، أعطِ "
                "مضاد حموضة وفق المصدر."
            ),
            source="D.2", evidence_codes=("heartburn",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="leg_cramps", value=True),
        salience=74,
    )
    def n24_leg_cramps(self):
        self._follow2_emit(
            rule_id="N24", severity="low", category="physiological_symptoms",
            rule_type="Treatment", priority="Medium", confidence="High",
            title="تدبير تشنجات الساق",
            detail=(
                "اعرض المغنيزيوم أو الكالسيوم أو خيارات علاجية غير دوائية حسب "
                "تفضيل المرأة."
            ),
            source="D.3", evidence_codes=("leg_cramps",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="preeclampsia_excluded", value=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {"low_back_pain", "pelvic_pain"}),
        salience=73,
    )
    def n25_back_or_pelvic_pain(self, trigger, code):
        self._follow2_emit(
            rule_id="N25", severity="low", category="physiological_symptoms",
            rule_type="Treatment", priority="Medium", confidence="High",
            title="ألم أسفل الظهر أو الحوض بعد استبعاد تسمم الحمل",
            detail=(
                "بعد استبعاد تسمم الحمل، أوصِ بتمارين منتظمة، واعرض العلاج الطبيعي "
                "أو أحزمة الدعم أو الوخز بالإبر حسب التفضيل."
            ),
            source="D.4", evidence_codes=(code, "preeclampsia_excluded"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="constipation", value=True),
        salience=72,
    )
    def n26_constipation(self):
        self._follow2_emit(
            rule_id="N26", severity="low", category="physiological_symptoms",
            rule_type="Treatment", priority="Medium", confidence="High",
            title="تدبير الإمساك",
            detail=(
                "أعطِ نخالة القمح أو مكملات ألياف أخرى إذا لم يستجب الإمساك "
                "للتعديل الغذائي وحده."
            ),
            source="D.5", evidence_codes=("constipation",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="preeclampsia_excluded", value=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {"varicose_veins", "limb_swelling", "edema"}),
        salience=71,
    )
    def n27_varicose_veins_or_swelling(self, trigger, code):
        self._follow2_emit(
            rule_id="N27", severity="low", category="physiological_symptoms",
            rule_type="Treatment", priority="Medium", confidence="High",
            title="دوالي الساقين أو تورم الأطراف بعد استبعاد تسمم الحمل",
            detail=(
                "بعد استبعاد تسمم الحمل، اعرض الجوارب الضاغطة، أو رفع الأطراف، "
                "أو الغمر بالماء."
            ),
            source="D.6", evidence_codes=(code, "preeclampsia_excluded"),
        )

    # ------------------------------------------------------------------
    # 10. Eight-contact model — N28
    # ------------------------------------------------------------------
    @Rule(PatientContext(pregnant=True), salience=31)
    def n28_eight_contact_model(self):
        self._follow2_emit(
            rule_id="N28", severity="low", category="antenatal_contact_schedule",
            rule_type="Monitoring", priority="High", confidence="High",
            title="اعتماد نموذج التماسات الثمانية",
            detail=(
                "اعتمد التماسات الحمل حتى الأسبوع 12، ثم الأسابيع 20 و26 و30 و34 "
                "و36 و38 و40؛ وإذا لم تحدث الولادة فالعودة في الأسبوع 41. هذه هي "
                "النسخة المعتمدة التي تحل محل أي نموذج قديم من أربع زيارات."
            ),
            source="E.7",
        )
