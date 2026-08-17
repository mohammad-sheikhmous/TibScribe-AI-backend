"""Pregnancy follow-up rules converted from ``1_kb_rules.md``.

Source basis:
    WHO PCPNC 2015 — Pregnancy, Childbirth, Postpartum and Newborn Care:
    A Guide for Essential Practice.

The rules in this module intentionally preserve the conditions and recommendations
written in the supplied Markdown source. They are decision-support suggestions for
clinician review, not autonomous treatment orders.
"""
from __future__ import annotations

from experta import AND, OR, AS, MATCH, NOT, Rule, TEST

from ..facts import ClinicalFact, PatientContext, RuleState
from ..suggestions import Suggestion

FOLLOW1_RULES_VERSION = "1.0.0"
SOURCE_NAME = "1_kb_rules.md — WHO PCPNC 2015"


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


class Follow1Rules:
    """Fifty atomic rules A1–A50 from the first pregnancy follow-up file."""

    def _follow1_evidence(self, codes: set[str] | tuple[str, ...] | list[str]) -> list[dict]:
        wanted = set(codes)
        evidence: list[dict] = []
        for fact in getattr(self, "facts", {}).values():
            if isinstance(fact, ClinicalFact) and fact.get("code") in wanted:
                evidence.append(_fact_evidence(fact))
        return evidence

    def _follow1_emit(
        self,
        *,
        rule_id: str,
        severity: str,
        category: str,
        title: str,
        detail: str,
        source: str,
        evidence_codes: tuple[str, ...] = (),
        chain: list[str] | None = None,
        missing_tests: list[str] | None = None,
        mark_state: bool = True,
    ) -> None:
        full_id = f"FOLLOW1-{rule_id}"
        evidence = self._follow1_evidence(evidence_codes)
        self._emit(
            Suggestion(
                rule_id=full_id,
                rule_version=FOLLOW1_RULES_VERSION,
                severity=severity,
                condition=category,
                title_ar=title,
                detail_ar=detail,
                missing_tests=missing_tests or [],
                evidence=evidence,
                chain=chain or [
                    f"تحققت شروط القاعدة {full_id}",
                    f"الفئة: {category}",
                    "إصدار توصية للمراجعة السريرية",
                ],
                references=[f"{SOURCE_NAME} — {rule_id} — {source}"],
            ),
            dedupe=(full_id, category),
        )
        if mark_state:
            self.declare(RuleState(code=full_id, category=category))

    # ------------------------------------------------------------------
    # 1. Basic delivery-place classification — A1..A3
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {
            "previous_cesarean", "transverse_lie_near_term", "multiple_pregnancy",
            "maternal_age_under_14", "previous_third_degree_tear",
            "vaginal_bleeding", "current_pregnancy_complication",
        }),
        salience=100,
    )
    def a1_referral_delivery_site(self, trigger, code):
        self._follow1_emit(
            rule_id="A1", severity="high", category="delivery_site_classification",
            title="مكان الولادة: مستشفى مرجعي",
            detail=(
                "صنّف مكان الولادة كمستوى إحالة (مستشفى مرجعي)، واشرح للمريضة "
                "سبب الضرورة، وأعد خطة الولادة والطوارئ."
            ),
            source="كتاب1 C2",
            evidence_codes=(code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        NOT(RuleState(code="FOLLOW1-A1")),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {
            "primipara", "previous_neonatal_day1_death", "maternal_age_under_16",
            "parity_over_6", "previous_labour_bleeding", "previous_labour_convulsions",
            "previous_instrumental_delivery", "hiv_positive",
        }),
        salience=90,
    )
    def a2_primary_care_delivery_site(self, trigger, code):
        self._follow1_emit(
            rule_id="A2", severity="medium", category="delivery_site_classification",
            title="مكان الولادة: رعاية أولية على الأقل",
            detail=(
                "صنّف مكان الولادة كمستوى رعاية أولية على الأقل، واشرح السبب، "
                "وأعد خطة الولادة والطوارئ."
            ),
            source="كتاب1 C2",
            evidence_codes=(code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="delivery_site_assessment_complete", value=True),
        NOT(RuleState(code="FOLLOW1-A1")),
        NOT(RuleState(code="FOLLOW1-A2")),
        salience=1,
    )
    def a3_delivery_site_by_preference(self):
        self._follow1_emit(
            rule_id="A3", severity="low", category="delivery_site_classification",
            title="مكان الولادة وفق رغبة المريضة",
            detail=(
                "عند غياب مؤشرات الإحالة والرعاية الأولية، اترك مكان الولادة لرغبة "
                "المريضة، مع تفضيل حضور قابلة أو طبيب مؤهل."
            ),
            source="كتاب1 C2",
        )

    # ------------------------------------------------------------------
    # 2. Pre-eclampsia / eclampsia — A4..A9
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        OR(
            AND(
                ClinicalFact(code="diastolic_bp_ge_110", value=True),
                ClinicalFact(code="proteinuria_ge_3", value=True),
            ),
            AND(
                ClinicalFact(code="diastolic_bp_ge_90", value=True),
                ClinicalFact(code="bp_readings_at_least_2", value=True),
                ClinicalFact(code="proteinuria_ge_2", value=True),
                OR(
                    ClinicalFact(code="severe_headache", value=True),
                    ClinicalFact(code="blurred_vision", value=True),
                    ClinicalFact(code="epigastric_pain", value=True),
                ),
            ),
        ),
        salience=120,
    )
    def a4_severe_preeclampsia(self):
        self._follow1_emit(
            rule_id="A4", severity="critical", category="preeclampsia",
            title="تسمم حمل شديد",
            detail=(
                "شخّص تسمم حمل شديداً؛ أعطِ كبريتات المغنيزيوم وخافض ضغط وريدياً، "
                "وعدّل خطة الولادة فوراً لمستشفى مرجعي، وحوّل عاجلاً جداً وفق المصدر."
            ),
            source="كتاب1 C3",
            evidence_codes=(
                "diastolic_bp", "proteinuria_grade", "bp_readings_at_least_2",
                "severe_headache", "blurred_vision", "epigastric_pain",
            ),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="diastolic_bp_90_to_110", value=True),
        ClinicalFact(code="bp_readings_at_least_2", value=True),
        ClinicalFact(code="proteinuria_ge_2", value=True),
        NOT(RuleState(code="FOLLOW1-A4")),
        salience=110,
    )
    def a5_preeclampsia(self):
        self._follow1_emit(
            rule_id="A5", severity="high", category="preeclampsia",
            title="تسمم حمل",
            detail="شخّص تسمم حمل، وعدّل خطة الولادة، وحوّل للمستشفى للمتابعة.",
            source="كتاب1 C3",
            evidence_codes=("diastolic_bp", "proteinuria_grade", "bp_readings_at_least_2"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="diastolic_bp_ge_90", value=True),
        ClinicalFact(code="bp_readings_at_least_2", value=True),
        ClinicalFact(code="proteinuria_lt_2", value=True),
        NOT(RuleState(code="FOLLOW1-A4")),
        NOT(RuleState(code="FOLLOW1-A5")),
        salience=100,
    )
    def a6_gestational_hypertension(self):
        self._follow1_emit(
            rule_id="A6", severity="medium", category="gestational_hypertension",
            title="ارتفاع ضغط دم الحمل",
            detail=(
                "شخّص ارتفاع ضغط دم الحمل؛ خفّف حمل العمل، وشجّع الراحة، "
                "وعلّم المريضة علامات الخطر."
            ),
            source="كتاب1 C3",
            evidence_codes=("diastolic_bp", "proteinuria_grade", "bp_readings_at_least_2"),
        )

    @Rule(
        RuleState(code="FOLLOW1-A6"),
        ClinicalFact(code="gestation_more_than_8_months", value=True),
        salience=80,
    )
    def a7_hypertension_week_followup(self):
        self._follow1_emit(
            rule_id="A7", severity="medium", category="gestational_hypertension_monitoring",
            title="إعادة تقييم ضغط الحمل خلال أسبوع",
            detail=(
                "أعد التقييم بعد أسبوع واحد؛ إن استمر الارتفاع بعد أسبوع، "
                "حوّل للمستشفى أو استشر طبيباً."
            ),
            source="كتاب1 C3",
            evidence_codes=("gestation_more_than_8_months",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="preeclampsia_assessment_complete", value=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {
            "previous_preeclampsia", "previous_eclampsia", "multiple_pregnancy",
            "chronic_hypertension", "chronic_kidney_disease", "autoimmune_disease",
        }),
        NOT(RuleState(code="FOLLOW1-A4")),
        NOT(RuleState(code="FOLLOW1-A5")),
        NOT(RuleState(code="FOLLOW1-A6")),
        salience=70,
    )
    def a8_preeclampsia_risk_prevention(self, trigger, code):
        self._follow1_emit(
            rule_id="A8", severity="medium", category="preeclampsia_risk",
            title="خطر تسمم الحمل",
            detail=(
                "صنّف الحالة ضمن فئة خطر تسمم الحمل؛ أعطِ أسبرين وقائياً، "
                "وكالسيوم إذا كان الاستهلاك الغذائي منخفضاً، وفق المصدر."
            ),
            source="كتاب1 C3",
            evidence_codes=(code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="preeclampsia_assessment_complete", value=True),
        NOT(RuleState(code="FOLLOW1-A4")),
        NOT(RuleState(code="FOLLOW1-A5")),
        NOT(RuleState(code="FOLLOW1-A6")),
        NOT(RuleState(code="FOLLOW1-A7")),
        NOT(RuleState(code="FOLLOW1-A8")),
        salience=2,
    )
    def a9_no_hypertension(self):
        self._follow1_emit(
            rule_id="A9", severity="low", category="preeclampsia",
            title="لا يوجد ارتفاع ضغط وفق القواعد المتاحة",
            detail="لم تتحقق قواعد تسمم الحمل أو ارتفاع ضغط الحمل السابقة.",
            source="كتاب1 C3",
        )

    # ------------------------------------------------------------------
    # 3. Anaemia in pregnancy — A10..A12
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        OR(
            ClinicalFact(code="hemoglobin_lt_7", value=True),
            ClinicalFact(code="severe_pallor", value=True),
            AND(
                ClinicalFact(code="pallor_present", value=True),
                OR(
                    ClinicalFact(code="respiratory_rate_gt_30", value=True),
                    ClinicalFact(code="easy_fatigue", value=True),
                    ClinicalFact(code="dyspnea_at_rest", value=True),
                ),
            ),
        ),
        salience=105,
    )
    def a10_severe_anemia(self):
        self._follow1_emit(
            rule_id="A10", severity="high", category="anemia",
            title="فقر دم شديد",
            detail=(
                "شخّص فقر دم شديداً؛ عدّل خطة الولادة لمركز يوفر نقل دم، وأعطِ "
                "حديداً مضاعفاً لمدة 3 أشهر، واتبع تدبير الملاريا المذكور في المصدر، "
                "وحوّل عاجلاً، وحدد متابعة خلال أسبوعين."
            ),
            source="كتاب1 C4",
            evidence_codes=(
                "hemoglobin_g_dl", "severe_pallor", "pallor_present",
                "respiratory_rate", "easy_fatigue", "dyspnea_at_rest",
            ),
        )

    @Rule(
        PatientContext(pregnant=True),
        OR(
            ClinicalFact(code="hemoglobin_7_to_11", value=True),
            ClinicalFact(code="pallor_present", value=True),
        ),
        NOT(RuleState(code="FOLLOW1-A10")),
        salience=95,
    )
    def a11_moderate_anemia(self):
        self._follow1_emit(
            rule_id="A11", severity="medium", category="anemia",
            title="فقر دم متوسط",
            detail=(
                "شخّص فقر دم متوسطاً؛ أعطِ حديداً مضاعفاً لمدة 3 أشهر، واتبع "
                "تدبير الملاريا المذكور في المصدر، وحدد إعادة تقييم خلال 4–6 أسابيع."
            ),
            source="كتاب1 C4",
            evidence_codes=("hemoglobin_g_dl", "pallor_present"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="hemoglobin_gt_11", value=True),
        ClinicalFact(code="pallor_present", value=False),
        salience=40,
    )
    def a12_anemia_prevention(self):
        self._follow1_emit(
            rule_id="A12", severity="low", category="anemia_prevention",
            title="لا يوجد فقر دم سريري",
            detail="أعطِ قرص حديد واحداً يومياً للوقاية لمدة 3 أشهر وفق المصدر.",
            source="كتاب1 C4",
            evidence_codes=("hemoglobin_g_dl", "pallor_present"),
        )

    # ------------------------------------------------------------------
    # 4. Syphilis screening — A13..A14
    # ------------------------------------------------------------------
    @Rule(PatientContext(pregnant=True), ClinicalFact(code="rpr_positive", value=True), salience=90)
    def a13_positive_rpr(self):
        self._follow1_emit(
            rule_id="A13", severity="high", category="syphilis",
            title="احتمال الإصابة بالزهري",
            detail=(
                "تحقق من حساسية البنسلين أولاً، ثم اتبع علاج الزهري المذكور في المصدر، "
                "وخطط لعلاج المولود عند الولادة، وشجّع فحص وعلاج الشريك."
            ),
            source="كتاب1 C5",
            evidence_codes=("rpr_positive",),
        )

    @Rule(PatientContext(pregnant=True), ClinicalFact(code="rpr_negative", value=True), salience=40)
    def a14_negative_rpr(self):
        self._follow1_emit(
            rule_id="A14", severity="low", category="syphilis_counseling",
            title="فحص الزهري سلبي",
            detail="أرشد حول الجنس الآمن للوقاية.",
            source="كتاب1 C5",
            evidence_codes=("rpr_negative",),
        )

    # ------------------------------------------------------------------
    # 5. HIV status — A15..A17
    # ------------------------------------------------------------------
    @Rule(PatientContext(pregnant=True), ClinicalFact(code="hiv_positive", value=True), salience=95)
    def a15_hiv_positive(self):
        self._follow1_emit(
            rule_id="A15", severity="high", category="hiv",
            title="إصابة بفيروس HIV",
            detail=(
                "ابدأ العلاج المضاد للفيروسات وفق البروتوكول، وأحِل لخدمات HIV "
                "المتخصصة، وقدّم دعماً نفسياً، وشجّع فحص الشريك، وحدّث تصنيف مكان "
                "الولادة لرعاية أولية على الأقل."
            ),
            source="كتاب1 C6",
            evidence_codes=("hiv_positive",),
        )

    @Rule(PatientContext(pregnant=True), ClinicalFact(code="hiv_negative", value=True), salience=45)
    def a16_hiv_negative(self):
        self._follow1_emit(
            rule_id="A16", severity="low", category="hiv_counseling",
            title="فحص HIV سلبي",
            detail=(
                "أرشد حول أهمية البقاء سلبية والجنس الآمن، وشجّع فحص الشريك، "
                "وحدد إعادة فحص HIV في الثلث الثالث من الحمل."
            ),
            source="كتاب1 C6",
            evidence_codes=("hiv_negative",),
        )

    @Rule(
        PatientContext(pregnant=True),
        OR(
            ClinicalFact(code="hiv_test_refused", value=True),
            ClinicalFact(code="hiv_status_unknown", value=True),
        ),
        salience=70,
    )
    def a17_hiv_unknown(self):
        self._follow1_emit(
            rule_id="A17", severity="medium", category="hiv_screening",
            title="حالة HIV غير معروفة",
            detail=(
                "قيّم علامات الإصابة الشديدة أو المتقدمة، وأرشد حول الجنس الآمن "
                "وفحص الشريك."
            ),
            source="كتاب1 C6",
            evidence_codes=("hiv_test_refused", "hiv_status_unknown"),
        )

    # ------------------------------------------------------------------
    # 6. Fetal movement — A18..A19
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="fetal_movement_absent", value=True),
        ClinicalFact(code="fetal_heartbeat_absent", value=True),
        ClinicalFact(code="rechecked_after_1h", value=True),
        salience=115,
    )
    def a18_possible_iufd(self):
        self._follow1_emit(
            rule_id="A18", severity="critical", category="fetal_movement",
            title="احتمال وفاة الجنين داخل الرحم",
            detail=(
                "أبلغ الأم والشريك بالاحتمال بحساسية، وحوّل فوراً للمستشفى وفق المصدر."
            ),
            source="كتاب1 C7",
            evidence_codes=("fetal_movement_absent", "fetal_heartbeat_absent", "rechecked_after_1h"),
        )

    @Rule(
        PatientContext(pregnant=True),
        OR(
            ClinicalFact(code="fetal_movement_absent", value=True),
            ClinicalFact(code="reduced_fetal_movement", value=True),
        ),
        ClinicalFact(code="fetal_heartbeat_present", value=True),
        NOT(RuleState(code="FOLLOW1-A18")),
        salience=85,
    )
    def a19_reduced_movement_heartbeat_present(self):
        self._follow1_emit(
            rule_id="A19", severity="low", category="fetal_movement",
            title="قلة حركة الجنين مع وجود نبض",
            detail=(
                "طمئن الأم أن الجنين على الأرجح بخير، وأوصِها بالعودة إذا استمرت المشكلة."
            ),
            source="كتاب1 C7",
            evidence_codes=("fetal_movement_absent", "reduced_fetal_movement", "fetal_heartbeat_present"),
        )

    # ------------------------------------------------------------------
    # 7. Rupture of membranes — A20..A22
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="rupture_of_membranes", value=True),
        ClinicalFact(code="temperature_ge_38", value=True),
        ClinicalFact(code="foul_vaginal_discharge", value=True),
        salience=115,
    )
    def a20_intrauterine_infection(self):
        self._follow1_emit(
            rule_id="A20", severity="critical", category="rupture_of_membranes",
            title="اشتباه التهاب رحمي وجنيني",
            detail=(
                "أعطِ مضاداً حيوياً وريدياً أو عضلياً فوراً، وحوّل عاجلاً جداً وفق المصدر."
            ),
            source="كتاب1 C7",
            evidence_codes=("rupture_of_membranes", "temperature_c", "foul_vaginal_discharge"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="rupture_of_membranes", value=True),
        ClinicalFact(code="gestation_less_than_8_months", value=True),
        ClinicalFact(code="rom_infection_assessment_complete", value=True),
        NOT(RuleState(code="FOLLOW1-A20")),
        salience=95,
    )
    def a21_preterm_rom(self):
        self._follow1_emit(
            rule_id="A21", severity="high", category="rupture_of_membranes",
            title="خطر التهاب رحمي وجنيني مع تمزق مبكر للأغشية",
            detail=(
                "اتبع تدبير الكورتيكوستيرويد عند عمر 24–34 أسبوعاً إذا كانت الولادة "
                "المبكرة متوقعة بلا دليل عدوى، وأعطِ الإريثرومايسين الوقائي، وحوّل عاجلاً."
            ),
            source="كتاب1 C7",
            evidence_codes=("rupture_of_membranes", "gestation_less_than_8_months", "gestation_24_to_34_weeks"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="rupture_of_membranes", value=True),
        ClinicalFact(code="gestation_8_months_or_more", value=True),
        ClinicalFact(code="rom_infection_assessment_complete", value=True),
        NOT(RuleState(code="FOLLOW1-A20")),
        salience=80,
    )
    def a22_term_rom(self):
        self._follow1_emit(
            rule_id="A22", severity="medium", category="rupture_of_membranes",
            title="تمزق الأغشية عند ثمانية أشهر أو أكثر",
            detail="أدِر الحالة كبداية مخاض وأحِلها إلى بروتوكول المخاض والولادة.",
            source="كتاب1 C7",
            evidence_codes=("rupture_of_membranes", "gestation_8_months_or_more"),
        )

    # ------------------------------------------------------------------
    # 8. Fever / UTI / malaria — A23..A26
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="temperature_gt_38", value=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {
            "very_rapid_breathing", "neck_stiffness", "lethargy",
            "partial_loss_of_consciousness", "unable_to_stand",
        }),
        salience=115,
    )
    def a23_very_severe_febrile_disease(self, trigger, code):
        self._follow1_emit(
            rule_id="A23", severity="critical", category="febrile_illness",
            title="مرض حموي شديد جداً",
            detail=(
                "أدخل خطاً وريدياً وأعطِ السوائل ببطء، واتبع المضاد الحيوي ومضاد "
                "الملاريا والغلوكوز المذكور في المصدر، وحوّل عاجلاً جداً."
            ),
            source="كتاب1 C8",
            evidence_codes=("temperature_c", code),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="temperature_gt_38", value=True),
        ClinicalFact(code="febrile_assessment_complete", value=True),
        OR(
            ClinicalFact(code="flank_pain", value=True),
            ClinicalFact(code="dysuria", value=True),
        ),
        NOT(RuleState(code="FOLLOW1-A23")),
        salience=100,
    )
    def a24_upper_uti(self):
        self._follow1_emit(
            rule_id="A24", severity="high", category="urinary_tract_infection",
            title="التهاب مسالك بولية علوي",
            detail=(
                "أعطِ مضاداً حيوياً ومضاد ملاريا فموياً وفق المصدر، وحوّل عاجلاً."
            ),
            source="كتاب1 C8",
            evidence_codes=("temperature_c", "flank_pain", "dysuria"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="febrile_assessment_complete", value=True),
        OR(
            ClinicalFact(code="temperature_gt_38", value=True),
            ClinicalFact(code="fever_last_48h", value=True),
        ),
        NOT(RuleState(code="FOLLOW1-A23")),
        NOT(RuleState(code="FOLLOW1-A24")),
        salience=80,
    )
    def a25_possible_malaria(self):
        self._follow1_emit(
            rule_id="A25", severity="medium", category="possible_malaria",
            title="ملاريا محتملة",
            detail=(
                "أكّد بفحص طفيلي مخبري، واتبع العلاج الفموي المذكور في المصدر، "
                "وحوّل إذا لم تتحسن خلال يومين."
            ),
            source="كتاب1 C8",
            evidence_codes=("temperature_c", "fever_last_48h"),
            missing_tests=["malaria_parasite_test"],
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="dysuria", value=True),
        ClinicalFact(code="temperature_not_over_38", value=True),
        ClinicalFact(code="febrile_assessment_complete", value=True),
        NOT(RuleState(code="FOLLOW1-A23")),
        NOT(RuleState(code="FOLLOW1-A24")),
        salience=75,
    )
    def a26_lower_uti(self):
        self._follow1_emit(
            rule_id="A26", severity="medium", category="urinary_tract_infection",
            title="التهاب مسالك بولية سفلي",
            detail=(
                "أعطِ مضاداً حيوياً فموياً، وشجّع شرب سوائل أكثر، وحوّل إذا لم "
                "تتحسن خلال يومين."
            ),
            source="كتاب1 C8",
            evidence_codes=("dysuria", "temperature_c"),
        )

    # ------------------------------------------------------------------
    # 9. Vaginal discharge — A27..A29
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="abnormal_vaginal_discharge", value=True),
        OR(
            ClinicalFact(code="partner_urethral_discharge", value=True),
            ClinicalFact(code="partner_dysuria", value=True),
        ),
        salience=90,
    )
    def a27_possible_gonorrhea_chlamydia(self):
        self._follow1_emit(
            rule_id="A27", severity="medium", category="vaginal_discharge",
            title="احتمال سيلان أو كلاميديا",
            detail=(
                "اتبع العلاج الفموي للمريضة والشريك معاً كما ورد في المصدر، "
                "وأرشد حول الجنس الآمن."
            ),
            source="كتاب1 C9",
            evidence_codes=("abnormal_vaginal_discharge", "partner_urethral_discharge", "partner_dysuria"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="curd_like_vaginal_discharge", value=True),
        ClinicalFact(code="severe_vulvar_itching", value=True),
        salience=85,
    )
    def a28_possible_candida(self):
        self._follow1_emit(
            rule_id="A28", severity="low", category="vaginal_discharge",
            title="احتمال عدوى فطرية (كانديدا)",
            detail="اتبع علاج الكلوتريمازول الموضعي المذكور في المصدر، وأرشد حول الجنس الآمن.",
            source="كتاب1 C9",
            evidence_codes=("curd_like_vaginal_discharge", "severe_vulvar_itching"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="abnormal_vaginal_discharge", value=True),
        ClinicalFact(code="vaginal_discharge_assessment_complete", value=True),
        NOT(RuleState(code="FOLLOW1-A27")),
        NOT(RuleState(code="FOLLOW1-A28")),
        salience=65,
    )
    def a29_possible_bv_or_trichomonas(self):
        self._follow1_emit(
            rule_id="A29", severity="low", category="vaginal_discharge",
            title="احتمال عدوى بكتيرية أو تريكوموناس",
            detail="اتبع علاج المترونيدازول المذكور في المصدر، وأرشد حول الجنس الآمن.",
            source="كتاب1 C9",
            evidence_codes=("abnormal_vaginal_discharge",),
        )

    # ------------------------------------------------------------------
    # 10. Possible advanced HIV — A30
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="hiv_status_unknown", value=True),
        OR(
            ClinicalFact(code="advanced_hiv_sign_count_ge_2", value=True),
            AND(
                ClinicalFact(code="advanced_hiv_sign_count_ge_1", value=True),
                ClinicalFact(code="hiv_risk_factor_count_ge_1", value=True),
            ),
        ),
        salience=92,
    )
    def a30_possible_advanced_hiv(self):
        self._follow1_emit(
            rule_id="A30", severity="high", category="advanced_hiv_screening",
            title="احتمال قوي لعدوى HIV شديدة أو متقدمة",
            detail="حوّل للمستشفى لمزيد من التقييم وفق المصدر.",
            source="كتاب1 C10",
            evidence_codes=(
                "hiv_status_unknown", "advanced_hiv_sign_count",
                "hiv_risk_factor_count",
            ),
        )

    # ------------------------------------------------------------------
    # 11. Substance use and violence — A31..A32
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {"tobacco_use", "alcohol_use", "drug_use"}),
        salience=65,
    )
    def a31_substance_use_counseling(self, trigger, code):
        self._follow1_emit(
            rule_id="A31", severity="medium", category="substance_use",
            title="إرشاد حول تعاطي المواد",
            detail=(
                "أرشد حول التوقف عن التدخين وتجنب التدخين السلبي، وأحِل لمقدمي "
                "رعاية متخصصين للكحول أو المخدرات عند الحاجة."
            ),
            source="كتاب1 C11",
            evidence_codes=(code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {"domestic_violence_history", "violence_exposure"}),
        salience=70,
    )
    def a32_violence_referral(self, trigger, code):
        self._follow1_emit(
            rule_id="A32", severity="medium", category="violence",
            title="الحاجة إلى استشارة متخصصة حول العنف",
            detail="قدّم استشارة متخصصة حول العنف وفق المصدر.",
            source="كتاب1 C11",
            evidence_codes=(code,),
        )

    # ------------------------------------------------------------------
    # 12. Cough and breathing difficulty — A33..A35
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="pneumonia_sign_count_ge_2", value=True),
        salience=95,
    )
    def a33_possible_pneumonia(self):
        self._follow1_emit(
            rule_id="A33", severity="high", category="respiratory",
            title="احتمال التهاب رئوي",
            detail="أعطِ أول جرعة مضاد حيوي وريدي أو عضلي، وحوّل عاجلاً وفق المصدر.",
            source="كتاب1 C11",
            evidence_codes=("temperature_c", "shortness_of_breath", "chest_pain"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="respiratory_assessment_complete", value=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {"cough_over_3_weeks", "hemoptysis", "wheeze"}),
        NOT(RuleState(code="FOLLOW1-A33")),
        salience=80,
    )
    def a34_possible_chronic_lung_disease(self, trigger, code):
        self._follow1_emit(
            rule_id="A34", severity="medium", category="respiratory",
            title="احتمال مرض رئوي مزمن",
            detail="حوّل للتقييم، وإذا كان الأزيز شديداً فحوّل عاجلاً وفق المصدر.",
            source="كتاب1 C11",
            evidence_codes=(code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="temperature_lt_38", value=True),
        ClinicalFact(code="cough_under_3_weeks", value=True),
        ClinicalFact(code="respiratory_assessment_complete", value=True),
        NOT(RuleState(code="FOLLOW1-A33")),
        NOT(RuleState(code="FOLLOW1-A34")),
        salience=50,
    )
    def a35_simple_upper_respiratory_infection(self):
        self._follow1_emit(
            rule_id="A35", severity="low", category="respiratory",
            title="التهاب تنفسي علوي بسيط",
            detail=(
                "صف علاجاً مهدئاً آمناً، وأرشد بالتوقف عن التدخين إن وُجد، وفق المصدر."
            ),
            source="كتاب1 C11",
            evidence_codes=("temperature_c", "cough_under_3_weeks"),
        )

    # ------------------------------------------------------------------
    # 13. Tuberculosis treatment — A36..A37
    # ------------------------------------------------------------------
    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="tb_treatment", value=True),
        ClinicalFact(code="streptomycin_in_treatment", value=True),
        salience=95,
    )
    def a36_tb_streptomycin(self):
        self._follow1_emit(
            rule_id="A36", severity="high", category="tuberculosis_treatment",
            title="مراجعة عاجلة لعلاج السل المحتوي على ستربتومايسين",
            detail=(
                "حوّل لمستشفى المديرية لمراجعة العلاج؛ يذكر المصدر أن الستربتومايسين "
                "سام لأذن الجنين."
            ),
            source="كتاب1 C11",
            evidence_codes=("tb_treatment", "streptomycin_in_treatment"),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="tb_treatment", value=True),
        ClinicalFact(code="streptomycin_in_treatment", value=False),
        salience=70,
    )
    def a37_tb_without_streptomycin(self):
        self._follow1_emit(
            rule_id="A37", severity="medium", category="tuberculosis_treatment",
            title="استمرار علاج السل دون ستربتومايسين",
            detail=(
                "طمئنها وفق المصدر وحثها على إكمال العلاج، وخطط لعلاج المولود وقائياً "
                "عند تحقق شروط المصدر، وعزّز إرشاد فحص HIV وافحص أفراد الأسرة المقربين."
            ),
            source="كتاب1 C11",
            evidence_codes=("tb_treatment",),
        )

    # ------------------------------------------------------------------
    # 14. Routine preventive measures — A38..A41
    # ------------------------------------------------------------------
    @Rule(PatientContext(pregnant=True), ClinicalFact(code="tetanus_vaccination_incomplete", value=True), salience=65)
    def a38_tetanus_vaccination(self):
        self._follow1_emit(
            rule_id="A38", severity="low", category="routine_prevention",
            title="استكمال لقاح التيتانوس",
            detail="أعطِ لقاح التيتانوس المستحق، وخطط للجرعة التالية في الزيارة القادمة.",
            source="كتاب1 C12",
            evidence_codes=("tetanus_vaccination_incomplete",),
        )

    @Rule(
        PatientContext(pregnant=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {
            "insufficient_iron_stock", "insufficient_folic_acid_stock",
            "insufficient_aspirin_stock", "insufficient_calcium_stock",
        }),
        salience=64,
    )
    def a39_supplement_stock(self, trigger, code):
        self._follow1_emit(
            rule_id="A39", severity="low", category="routine_prevention",
            title="توفير مخزون الأدوية الوقائية",
            detail=(
                "اصرف مخزون 3 أشهر من الحديد وحمض الفوليك، ومن الأسبرين أو الكالسيوم "
                "إذا كانا موصوفين، وأرشد حول الالتزام والسلامة."
            ),
            source="كتاب1 C12",
            evidence_codes=(code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="trimester_2_or_3", value=True),
        ClinicalFact(code="mebendazole_given", value=False),
        salience=55,
    )
    def a40_mebendazole(self):
        self._follow1_emit(
            rule_id="A40", severity="low", category="routine_prevention",
            title="جرعة ميبيندازول في الثلث الثاني أو الثالث",
            detail="أعطِ ميبيندازول مرة واحدة في الثلث الثاني أو الثالث وفق المصدر.",
            source="كتاب1 C12",
            evidence_codes=("trimester_2_or_3",),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="malaria_endemic_area", value=True),
        ClinicalFact(code="malaria_ipt_recently_given", value=False),
        salience=55,
    )
    def a41_malaria_prevention(self):
        self._follow1_emit(
            rule_id="A41", severity="low", category="routine_prevention",
            title="الوقاية المتقطعة من الملاريا",
            detail=(
                "أعطِ العلاج الوقائي المتقطع في الثلثين الثاني والثالث، وشجّع النوم "
                "تحت ناموسية مشبعة بمبيد وفق المصدر."
            ),
            source="كتاب1 C12",
            evidence_codes=("malaria_endemic_area",),
        )

    # ------------------------------------------------------------------
    # 15. Nutrition and self-care — A42..A43
    # ------------------------------------------------------------------
    @Rule(AS.context << PatientContext(pregnant=True), salience=20)
    def a42_pregnancy_nutrition_counseling(self, context):
        self._follow1_emit(
            rule_id="A42", severity="low", category="pregnancy_nutrition",
            title="إرشادات التغذية أثناء الحمل",
            detail=(
                "أرشد بزيادة تنوع وكمية الغذاء الصحي؛ وزد وقت الإرشاد إذا كانت "
                "المريضة نحيفة جداً أو مراهقة أو مصابة بـHIV، وصحّح المحرمات "
                "الغذائية الضارة."
            ),
            source="كتاب1 C13",
            chain=[
                "PatientContext.pregnant = True",
                (
                    f"عمر الحمل المسجل = {int(context.get('ga_weeks'))} أسبوع"
                    if context.get('ga_weeks')
                    else "عمر الحمل غير محدد في هذا التماس"
                ),
                "FOLLOW1-A42 → إرشادات التغذية أثناء الحمل",
            ],
        )

    @Rule(PatientContext(pregnant=True), salience=19)
    def a43_pregnancy_self_care(self):
        self._follow1_emit(
            rule_id="A43", severity="low", category="pregnancy_self_care",
            title="إرشادات الرعاية الذاتية أثناء الحمل",
            detail=(
                "أرشد بتناول أقراص الحديد، والراحة وتجنب حمل الأشياء الثقيلة، والنوم "
                "تحت ناموسية، والجنس الآمن عند وجود خطر، وتجنب الكحول والتدخين، "
                "وعدم تناول دواء غير موصوف."
            ),
            source="كتاب1 C13",
        )

    # ------------------------------------------------------------------
    # 16. Birth/emergency plan and danger signs — A44..A46
    # ------------------------------------------------------------------
    @Rule(PatientContext(pregnant=True), salience=18)
    def a44_birth_and_emergency_plan(self):
        self._follow1_emit(
            rule_id="A44", severity="low", category="birth_emergency_plan",
            title="خطة الولادة والطوارئ",
            detail=(
                "اشرح أسباب أفضلية الولادة داخل منشأة صحية، وراجع ترتيبات الوصول "
                "والتكلفة، وجهّز مستلزمات الولادة المذكورة في المصدر."
            ),
            source="كتاب1 C13-C14",
        )

    @Rule(
        PatientContext(pregnant=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {
            "vaginal_bleeding", "convulsions", "danger_headache_visual_cluster",
            "danger_fever_weakness_cluster", "severe_abdominal_pain",
            "shortness_of_breath", "rapid_breathing",
        }),
        salience=125,
    )
    def a45_immediate_danger_sign(self, trigger, code):
        self._follow1_emit(
            rule_id="A45", severity="critical", category="pregnancy_danger_signs",
            title="علامة خطر: التوجه فوراً إلى منشأة صحية",
            detail="وجّه المرأة للتوجه فوراً لمنشأة صحية دون انتظار.",
            source="كتاب1 C14",
            evidence_codes=(code,),
        )

    @Rule(
        PatientContext(pregnant=True),
        ClinicalFact(code="danger_sign_assessment_complete", value=True),
        AS.trigger << ClinicalFact(code=MATCH.code, value=True),
        TEST(lambda code: code in {
            "fever", "abdominal_pain", "generally_unwell",
            "swelling_fingers_face_legs",
        }),
        NOT(RuleState(code="FOLLOW1-A45")),
        salience=75,
    )
    def a46_soon_health_center(self, trigger, code):
        self._follow1_emit(
            rule_id="A46", severity="medium", category="pregnancy_warning_signs",
            title="مراجعة المركز الصحي في أقرب وقت",
            detail="وجّه المريضة للتوجه إلى المركز الصحي في أقرب وقت ممكن.",
            source="كتاب1 C14",
            evidence_codes=(code,),
        )

    # ------------------------------------------------------------------
    # 17. Follow-up scheduling — A47..A50
    # ------------------------------------------------------------------
    @Rule(
        RuleState(code="FOLLOW1-A6"),
        ClinicalFact(code="gestation_more_than_8_months", value=True),
        salience=68,
    )
    def a47_hypertension_followup_schedule(self):
        self._follow1_emit(
            rule_id="A47", severity="medium", category="follow_up_schedule",
            title="موعد متابعة خلال أسبوع",
            detail="حدد موعد إعادة الزيارة خلال أسبوع واحد.",
            source="كتاب1 C3",
            evidence_codes=("gestation_more_than_8_months",),
        )

    @Rule(RuleState(code="FOLLOW1-A10"), salience=67)
    def a48_severe_anemia_followup(self):
        self._follow1_emit(
            rule_id="A48", severity="high", category="follow_up_schedule",
            title="متابعة فقر الدم الشديد خلال أسبوعين",
            detail="حدد موعد إعادة الزيارة خلال أسبوعين.",
            source="كتاب1 C4",
        )

    @Rule(RuleState(code="FOLLOW1-A15"), salience=67)
    def a49_hiv_followup(self):
        self._follow1_emit(
            rule_id="A49", severity="medium", category="follow_up_schedule",
            title="متابعة HIV خلال أسبوعين",
            detail="حدد موعد إعادة الزيارة خلال أسبوعين بعد فحص HIV.",
            source="كتاب1 C6",
        )

    @Rule(
        PatientContext(pregnant=True),
        NOT(RuleState(code="FOLLOW2-N28")),
        salience=10,
    )
    def a50_eight_contact_schedule(self):
        self._follow1_emit(
            rule_id="A50", severity="low", category="antenatal_contact_schedule",
            title="جدول المتابعة بثمانية تماسات",
            detail=(
                "حدد جدول التماسات: الأول حتى الأسبوع 12، ثم الأسابيع 20، 26، 30، "
                "34، 36، 38، 40؛ وإذا لم تحدث الولادة فالعودة في الأسبوع 41. عند "
                "الإصابة بـHIV تأكد من تماس في الأسبوع 26–28. تبقى المواعيد الأقرب "
                "التي تفرضها قواعد الخطر نافذة."
            ),
            source="كتاب2 E.7 — كما ورد في ملف القواعد الأول",
        )
