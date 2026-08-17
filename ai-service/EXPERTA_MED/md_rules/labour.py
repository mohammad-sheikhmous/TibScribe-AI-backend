"""Labour rules converted from ``kb_rules_labour.md``.

Source basis:
    WHO PCPNC 2015 + WHO Intrapartum Care 2018, as structured in the supplied
    Markdown file. When the supplied file identifies a conflict, the 2018 source
    is treated as controlling.

The rules are clinical decision-support suggestions for clinician review. They
are not autonomous medical orders.
"""
from __future__ import annotations

from experta import AS, MATCH, Rule, TEST

from ..facts import ClinicalFact, RuleState
from ..suggestions import Suggestion

FOLLOW3_RULES_VERSION = "1.0.0"
SOURCE_NAME = "kb_rules_labour.md — WHO PCPNC 2015 + WHO Intrapartum Care 2018"


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


class Follow3Rules:
    """Forty-one atomic labour rules L1–L41 from the third source file."""

    def _follow3_evidence(self, codes) -> list[dict]:
        wanted = set(codes)
        evidence: list[dict] = []
        for fact in getattr(self, "facts", {}).values():
            if isinstance(fact, ClinicalFact) and fact.get("code") in wanted:
                evidence.append(_fact_evidence(fact))
        return evidence

    def _follow3_emit(
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
        evidence_codes=(),
        chain: list[str] | None = None,
    ) -> None:
        full_id = f"FOLLOW3-{rule_id}"
        evidence = self._follow3_evidence(evidence_codes)
        default_chain = [
            f"تحققت شروط القاعدة {full_id}",
            f"الفئة: {category}؛ النوع: {rule_type}",
            f"الأولوية: {priority}؛ قوة الدليل: {confidence}",
            "إصدار توصية للمراجعة السريرية",
        ]
        self._emit(
            Suggestion(
                rule_id=full_id,
                rule_version=FOLLOW3_RULES_VERSION,
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
        self.declare(RuleState(code=full_id, category=category))

    # ==================================================================
    # 1. Admission assessment and labour stage — L1..L10
    # ==================================================================
    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="spontaneous_labour", value=True),
        salience=70,
    )
    def l1_admission_assessment(self):
        self._follow3_emit(
            rule_id="L1", severity="low", category="labour_admission",
            rule_type="Assessment", priority="High", confidence="High",
            title="تقييم الدخول الكامل عند بدء المخاض",
            detail=(
                "خذ تاريخ بدء الانقباضات وتكرارها وحالة الأغشية والنزيف وحركة الجنين، "
                "وراجع سجل الحمل، وافحص البطن، واستمع لنبض الجنين دقيقة كاملة، "
                "وقس الضغط والحرارة وافحص الشحوب والجفاف."
            ),
            source="كتاب1 D2",
            evidence_codes=("in_labour", "spontaneous_labour"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="current_vaginal_bleeding", value=False),
        ClinicalFact(code="bleeding_after_seventh_month", value=False),
        salience=88,
    )
    def l2_vaginal_exam_allowed(self):
        self._follow3_emit(
            rule_id="L2", severity="low", category="labour_admission",
            rule_type="Assessment", priority="High", confidence="High",
            title="الفحص المهبلي بعد استبعاد النزيف",
            detail=(
                "اشرح الفحص المهبلي وخذ الموافقة، ولا تحلق العجان، وحدد اتساع "
                "عنق الرحم بالسنتيمتر لتصنيف مرحلة المخاض."
            ),
            source="كتاب1 D2 / كتاب3 rec.5",
            evidence_codes=("current_vaginal_bleeding", "bleeding_after_seventh_month"),
        )

    def _emit_l3(self, code: str) -> None:
        self._follow3_emit(
            rule_id="L3", severity="high", category="labour_admission",
            rule_type="Assessment", priority="High", confidence="High",
            title="منع الفحص المهبلي عند وجود نزيف",
            detail=(
                "لا تُجرِ الفحص المهبلي حتى يُستبعد احتمال المشيمة المنزاحة عند "
                "وجود نزيف حالي أو نزيف بعد الشهر السابع."
            ),
            source="كتاب1 D2", evidence_codes=(code,),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="current_vaginal_bleeding", value=True),
        salience=115,
    )
    def l3_current_bleeding(self):
        self._emit_l3("current_vaginal_bleeding")

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="bleeding_after_seventh_month", value=True),
        salience=115,
    )
    def l3_late_pregnancy_bleeding_history(self):
        self._emit_l3("bleeding_after_seventh_month")

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_latent", value=True),
        salience=82,
    )
    def l4_latent_stage(self):
        self._follow3_emit(
            rule_id="L4", severity="low", category="labour_stage",
            rule_type="Assessment", priority="Medium", confidence="High",
            title="تصنيف المرحلة الكامنة",
            detail="صنّف الحالة كمرحلة كامنة؛ لم تدخل المخاض النشط بعد.",
            source="كتاب3 rec.5",
            evidence_codes=("cervical_dilation_cm", "irregular_contractions", "labour_stage_latent"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_active", value=True),
        salience=84,
    )
    def l5_active_stage(self):
        self._follow3_emit(
            rule_id="L5", severity="low", category="labour_stage",
            rule_type="Assessment", priority="Medium", confidence="High",
            title="تصنيف المرحلة النشطة",
            detail="صنّف الحالة كمرحلة نشطة وابدأ مراقبة المرحلة النشطة وفق L33.",
            source="كتاب3 rec.5",
            evidence_codes=("cervical_dilation_cm", "labour_stage_active"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_second", value=True),
        salience=105,
    )
    def l6_second_stage(self):
        self._follow3_emit(
            rule_id="L6", severity="low", category="labour_stage",
            rule_type="Assessment", priority="High", confidence="High",
            title="بدء المرحلة الثانية من المخاض",
            detail=(
                "صنّف الحالة كبداية المرحلة الثانية، وسلّم المتابعة إلى ملف قواعد الولادة."
            ),
            source="كتاب3 rec.5",
            evidence_codes=("cervical_dilation_cm", "perineum_bulging", "fetal_head_visible", "labour_stage_second"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="healthy_labour_woman", value=True),
        ClinicalFact(code="spontaneous_labour", value=True),
        ClinicalFact(code="consider_delayed_labour_ward_admission", value=True),
        salience=60,
    )
    def l7_no_routine_delayed_admission(self):
        self._follow3_emit(
            rule_id="L7", severity="low", category="labour_admission",
            rule_type="Prevention", priority="Low", confidence="Low",
            title="عدم التوصية الروتينية بتأخير القبول",
            detail=(
                "لا تُوصِ روتينياً بتأخير قبول المرأة في جناح الولادة حتى المرحلة "
                "النشطة؛ يقتصر ذلك على سياق بحثي صارم وفق المصدر."
            ),
            source="كتاب3 rec.10",
            evidence_codes=("healthy_labour_woman", "spontaneous_labour", "consider_delayed_labour_ward_admission"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="healthy_labour_woman", value=True),
        salience=35,
    )
    def l8_no_routine_pelvimetry(self):
        self._follow3_emit(
            rule_id="L8", severity="low", category="labour_admission",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم إجراء قياس الحوض السريري روتينياً",
            detail="لا تُوصِ بقياس الحوض السريري الروتيني عند قبول امرأة سليمة للمخاض.",
            source="كتاب3 rec.11", evidence_codes=("healthy_labour_woman",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="healthy_labour_woman", value=True),
        ClinicalFact(code="spontaneous_labour", value=True),
        ClinicalFact(code="fetal_wellbeing_assessment_needed", value=True),
        salience=78,
    )
    def l9_intermit_fetal_heart_assessment(self):
        self._follow3_emit(
            rule_id="L9", severity="low", category="fetal_wellbeing",
            rule_type="Assessment", priority="High", confidence="High",
            title="تقييم نبض الجنين بالدوبلر أو سماعة بينارد",
            detail="استمع لنبض الجنين بجهاز دوبلر أو سماعة بينارد عند القبول.",
            source="كتاب3 rec.12",
            evidence_codes=("healthy_labour_woman", "spontaneous_labour", "fetal_wellbeing_assessment_needed"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="healthy_labour_woman", value=True),
        ClinicalFact(code="spontaneous_labour", value=True),
        salience=36,
    )
    def l10_no_routine_admission_ctg(self):
        self._follow3_emit(
            rule_id="L10", severity="low", category="fetal_wellbeing",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم التوصية بـCTG روتيني عند القبول",
            detail="لا تُوصِ بتخطيط قلب الجنين الروتيني عند قبول امرأة سليمة بمخاض عفوي.",
            source="كتاب3 rec.13",
            evidence_codes=("healthy_labour_woman", "spontaneous_labour"),
        )

    # ==================================================================
    # 2. Obstetric problems — L11..L21
    # ==================================================================
    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="obstructed_labour_sign", value=True),
        salience=130,
    )
    def l11_obstructed_labour(self):
        self._follow3_emit(
            rule_id="L11", severity="critical", category="obstructed_labour",
            rule_type="Diagnosis", priority="High", confidence="High",
            title="مخاض معاق — تحويل عاجل جداً",
            detail=(
                "شخّص مخاضاً معاقاً؛ ابدأ خطاً وريدياً وسوائل عند التعب، وأعطِ "
                "مضاداً حيوياً إذا تجاوز المخاض 24 ساعة، وحوّل عاجلاً جداً للمستشفى."
            ),
            source="كتاب1 D3",
            evidence_codes=(
                "transverse_lie", "continuous_contractions_no_rest",
                "constant_pain_between_contractions", "sudden_severe_abdominal_pain",
                "horizontal_abdominal_ridge", "labour_over_24h", "obstructed_labour_sign",
            ),
        )

    def _emit_l12(self, code: str) -> None:
        self._follow3_emit(
            rule_id="L12", severity="critical", category="intrauterine_infection",
            rule_type="Diagnosis", priority="High", confidence="High",
            title="التهاب رحمي وجنيني مع تمزق الأغشية",
            detail=(
                "شخّص التهاباً رحمياً وجنينياً؛ أعطِ مضاداً حيوياً فوراً، وأكمل "
                "الولادة إن كانت قريبة ثم حوّل، وخطط لعلاج المولود."
            ),
            source="كتاب1 D3",
            evidence_codes=("rupture_of_membranes", code),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="rupture_of_membranes", value=True),
        ClinicalFact(code="temperature_gt_38", value=True),
        salience=128,
    )
    def l12_rom_with_fever(self):
        self._emit_l12("temperature_gt_38")

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="rupture_of_membranes", value=True),
        ClinicalFact(code="foul_vaginal_discharge", value=True),
        salience=128,
    )
    def l12_rom_with_foul_discharge(self):
        self._emit_l12("foul_vaginal_discharge")

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="rupture_of_membranes", value=True),
        ClinicalFact(code="gestation_less_than_8_months", value=True),
        salience=104,
    )
    def l13_preterm_rom_prophylaxis(self):
        self._follow3_emit(
            rule_id="L13", severity="high", category="preterm_rom",
            rule_type="Treatment", priority="High", confidence="High",
            title="وقاية عند تمزق الأغشية قبل ثمانية أشهر",
            detail=(
                "أعطِ مضاداً حيوياً وقائياً، وخطط لعلاج المولود احتياطاً لاحتمال الخداج."
            ),
            source="كتاب1 D3",
            evidence_codes=("rupture_of_membranes", "gestation_less_than_8_months"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="diastolic_bp_gt_90", value=True),
        salience=108,
    )
    def l14_high_diastolic_bp_in_labour(self):
        self._follow3_emit(
            rule_id="L14", severity="high", category="hypertension_in_labour",
            rule_type="Assessment", priority="High", confidence="High",
            title="ارتفاع الضغط الانبساطي أثناء المخاض",
            detail=(
                "راقب الضغط والبروتين عن كثب، وأدر الحالة حسب شدة تسمم الحمل، "
                "بما في ذلك كبريتات المغنيزيوم عند الحالة الشديدة وفق المصدر."
            ),
            source="كتاب1 D3-D4",
            evidence_codes=("diastolic_bp", "diastolic_bp_gt_90"),
        )

    def _emit_l15(self, code: str) -> None:
        self._follow3_emit(
            rule_id="L15", severity="high", category="severe_anemia_in_labour",
            rule_type="Referral", priority="High", confidence="High",
            title="فقر دم شديد أثناء المخاض",
            detail="حوّل عاجلاً إلى منشأة تتوفر فيها إمكانية نقل الدم.",
            source="كتاب1 D3-D4", evidence_codes=(code,),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="severe_pallor", value=True),
        salience=112,
    )
    def l15_severe_pallor(self):
        self._emit_l15("severe_pallor")

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="hemoglobin_lt_7", value=True),
        salience=112,
    )
    def l15_low_hemoglobin(self):
        self._emit_l15("hemoglobin_lt_7")

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="preterm_labour", value=True),
        ClinicalFact(code="gestation_less_than_32_weeks", value=True),
        salience=110,
    )
    def l16_magnesium_for_fetal_neuroprotection(self):
        self._follow3_emit(
            rule_id="L16", severity="high", category="preterm_labour",
            rule_type="Medication", priority="High", confidence="High",
            title="كبريتات المغنيزيوم لحماية دماغ الجنين",
            detail=(
                "عند المخاض قبل إتمام ثمانية أشهر وعمر حمل أقل من 32 أسبوعاً، "
                "أعطِ كبريتات المغنيزيوم للحماية العصبية للجنين."
            ),
            source="كتاب1 D4",
            evidence_codes=("preterm_labour", "gestation_less_than_32_weeks"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="preterm_labour", value=True),
        salience=100,
    )
    def l17_preterm_birth_preparation(self):
        self._follow3_emit(
            rule_id="L17", severity="high", category="preterm_labour",
            rule_type="Treatment", priority="High", confidence="High",
            title="الاستعداد للولادة المبكرة",
            detail=(
                "تحكم بعناية بولادة الرأس وجهّز معدات إنعاش المولود، ولا تُوصِ "
                "بقيصرية روتينية لمجرد تحسين نتائج الخداج."
            ),
            source="كتاب1 D4", evidence_codes=("preterm_labour",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="fetal_heart_rate_abnormal", value=True),
        salience=122,
    )
    def l18_abnormal_fetal_heart_rate(self):
        self._follow3_emit(
            rule_id="L18", severity="high", category="fetal_distress",
            rule_type="Treatment", priority="High", confidence="High",
            title="نبض جنيني غير طبيعي",
            detail=(
                "غيّر وضعية الأم إلى الجانب الأيسر، وأوقف الأوكسيتوسين إن كان "
                "يُعطى، وأعد الفحص خلال 15–30 دقيقة، وحوّل إذا استمر الشذوذ."
            ),
            source="كتاب1 D4",
            evidence_codes=("fetal_heart_rate_bpm", "fetal_heart_rate_abnormal"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="fetal_movement_absent", value=True),
        ClinicalFact(code="fetal_heartbeat_absent", value=True),
        ClinicalFact(code="repeated_fetal_checks", value=True),
        salience=135,
    )
    def l19_probable_fetal_death(self):
        self._follow3_emit(
            rule_id="L19", severity="critical", category="probable_fetal_death",
            rule_type="Counseling", priority="High", confidence="High",
            title="اشتباه وفاة الجنين",
            detail="أبلغ الأهل بحساسية أن الجنين على الأرجح متوفى بعد الفحوصات المتكررة.",
            source="كتاب1 D4",
            evidence_codes=("fetal_movement_absent", "fetal_heartbeat_absent", "repeated_fetal_checks"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="dehydration_sign_count_ge_2", value=True),
        salience=90,
    )
    def l20_dehydration(self):
        self._follow3_emit(
            rule_id="L20", severity="medium", category="dehydration",
            rule_type="Diagnosis", priority="Medium", confidence="High",
            title="الجفاف أثناء المخاض",
            detail=(
                "شخّص الجفاف عند وجود علامتين أو أكثر؛ أعطِ سوائل فموية، أو لتراً "
                "وريدياً خلال ثلاث ساعات إذا تعذر الشرب."
            ),
            source="كتاب1 D4",
            evidence_codes=("thirst", "sunken_eyes", "dry_mouth", "slow_skin_pinch_return", "dehydration_sign_count_ge_2"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="delivery_extra_assistance_risk", value=True),
        salience=86,
    )
    def l21_prepare_extra_delivery_assistance(self):
        self._follow3_emit(
            rule_id="L21", severity="medium", category="delivery_preparation",
            rule_type="Assessment", priority="Medium", confidence="High",
            title="الحاجة إلى مساعدة إضافية عند الولادة",
            detail=(
                "سجّل الحاجة إلى مساعدة إضافية وتجهيز خاص عند اقتراب الولادة الفعلية، "
                "بما في ذلك التحضير لبضع فرج سخي وفق الحالة المذكورة في المصدر."
            ),
            source="كتاب1 D4",
            evidence_codes=(
                "genital_warts_or_scars_obstructing", "previous_third_degree_tear",
                "third_trimester_bleeding_history", "previous_cesarean",
                "previous_forceps_delivery", "maternal_age_under_14",
                "delivery_extra_assistance_risk",
            ),
        )

    # ==================================================================
    # 3. Respectful supportive care — L22..L29
    # ==================================================================
    @Rule(ClinicalFact(code="in_labour", value=True), salience=32)
    def l22_respectful_maternity_care(self):
        self._follow3_emit(
            rule_id="L22", severity="low", category="supportive_labour_care",
            rule_type="Counseling", priority="High", confidence="High",
            title="الرعاية المحترمة أثناء المخاض",
            detail=(
                "حافظ على الكرامة والخصوصية والسرية، وامنع الأذى وسوء المعاملة، "
                "وادعم الاختيار المستنير والدعم المستمر."
            ),
            source="كتاب3 rec.1", evidence_codes=("in_labour",),
        )

    @Rule(ClinicalFact(code="in_labour", value=True), salience=31)
    def l23_communication_and_consent(self):
        self._follow3_emit(
            rule_id="L23", severity="low", category="supportive_labour_care",
            rule_type="Counseling", priority="High", confidence="High",
            title="التواصل والموافقة أثناء المخاض",
            detail=(
                "اشرح كل إجراء واطلب الإذن وناقش النتائج، وأبقِ المرأة مطلعة على "
                "تقدم المخاض وشجعها وطمئنها، واحترم رغبتها في مشاركة معلومات HIV."
            ),
            source="كتاب3 rec.2 / كتاب1 D5", evidence_codes=("in_labour",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="wants_labour_companion", value=True),
        salience=72,
    )
    def l24_companion_of_choice(self):
        self._follow3_emit(
            rule_id="L24", severity="low", category="supportive_labour_care",
            rule_type="Counseling", priority="Medium", confidence="High",
            title="مرافق من اختيار المرأة",
            detail=(
                "اسمح بمرافق من اختيارها طوال المخاض، ووجّهه للتشجيع وطلب المساعدة "
                "فوراً عند ظهور علامة خطر دون التدخل المخالف للطاقم."
            ),
            source="كتاب3 rec.3 / كتاب1 D6", evidence_codes=("wants_labour_companion",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="midwife_continuity_program_available", value=True),
        salience=55,
    )
    def l25_midwife_continuity_model(self):
        self._follow3_emit(
            rule_id="L25", severity="low", category="supportive_labour_care",
            rule_type="Prevention", priority="Medium", confidence="Medium",
            title="نموذج استمرارية الرعاية بواسطة القابلات",
            detail=(
                "عند توفر برنامج فعّال، اعتمد قابلة معروفة أو مجموعة صغيرة تتابع "
                "المرأة خلال الحمل والمخاض والنفاس."
            ),
            source="كتاب3 rec.4", evidence_codes=("midwife_continuity_program_available",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="low_risk_labour", value=True),
        salience=46,
    )
    def l26_mobility_food_and_comfort(self):
        self._follow3_emit(
            rule_id="L26", severity="low", category="supportive_labour_care",
            rule_type="Counseling", priority="Medium", confidence="High",
            title="الحركة والراحة والأكل والشرب في المخاض منخفض الخطورة",
            detail=(
                "شجع الاستحمام والحركة والمشي والوضعية المفضلة وتفريغ المثانة كل "
                "ساعتين، واسمح بالأكل والشرب حسب رغبة المرأة."
            ),
            source="كتاب1 D5-D6 / كتاب3 rec.24-25", evidence_codes=("low_risk_labour",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="low_risk_labour", value=True),
        salience=45,
    )
    def l27_no_routine_enema_or_chlorhexidine(self):
        self._follow3_emit(
            rule_id="L27", severity="low", category="supportive_labour_care",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم التوصية بالحقنة الشرجية أو التنظيف المهبلي روتينياً",
            detail="لا تُوصِ بحقنة شرجية روتينية ولا تنظيف مهبلي روتيني بالكلورهيكسيدين.",
            source="كتاب3 rec.26", evidence_codes=("low_risk_labour",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="requests_labour_pain_relief", value=True),
        salience=74,
    )
    def l28_pain_relief_options(self):
        self._follow3_emit(
            rule_id="L28", severity="low", category="labour_pain_relief",
            rule_type="Counseling", priority="Medium", confidence="High",
            title="خيارات تسكين الألم أثناء المخاض",
            detail=(
                "اعرض حسب التفضيل التخدير فوق الجافية، أو الأفيونات الحقنية، أو "
                "تقنيات الاسترخاء، أو التدليك والكمادات الدافئة."
            ),
            source="كتاب3 rec.19-22", evidence_codes=("requests_labour_pain_relief",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="pain_relief_goal_accelerate_labour", value=True),
        salience=58,
    )
    def l29_no_analgesia_to_accelerate_labour(self):
        self._follow3_emit(
            rule_id="L29", severity="low", category="labour_pain_relief",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم استخدام التسكين بهدف تسريع المخاض",
            detail="لا تُوصِ بتسكين الألم كإجراء وقائي لمنع تأخر المخاض عندما لا يكون الهدف راحة المرأة.",
            source="كتاب3 rec.23", evidence_codes=("pain_relief_goal_accelerate_labour",),
        )

    # ==================================================================
    # 4. First-stage monitoring — L30..L35
    # ==================================================================
    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_latent", value=True),
        salience=65,
    )
    def l30_latent_stage_monitoring(self):
        self._follow3_emit(
            rule_id="L30", severity="low", category="labour_monitoring",
            rule_type="Monitoring", priority="High", confidence="High",
            title="مراقبة المرحلة الكامنة",
            detail=(
                "راقب علامات الطوارئ والانقباضات ونبض الجنين والمزاج كل ساعة، "
                "واتساع عنق الرحم والحرارة والنبض والضغط كل أربع ساعات."
            ),
            source="كتاب1 D7", evidence_codes=("labour_stage_latent",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_latent", value=True),
        AS.duration << ClinicalFact(code="latent_stage_duration_hours", value=MATCH.hours),
        TEST(lambda hours: float(hours) >= 8),
        ClinicalFact(code="contractions_strengthened", value=True),
        ClinicalFact(code="no_cervical_progress", value=True),
        salience=106,
    )
    def l31_prolonged_latent_stronger_no_progress(self, duration, hours):
        self._follow3_emit(
            rule_id="L31", severity="high", category="labour_progress",
            rule_type="Referral", priority="High", confidence="High",
            title="مرحلة كامنة مطولة مع انقباضات أقوى دون تقدم",
            detail="حوّل عاجلاً إلى المستشفى.",
            source="كتاب1 D7",
            evidence_codes=("labour_stage_latent", "latent_stage_duration_hours", "contractions_strengthened", "no_cervical_progress"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_latent", value=True),
        AS.duration << ClinicalFact(code="latent_stage_duration_hours", value=MATCH.hours),
        TEST(lambda hours: float(hours) >= 8),
        ClinicalFact(code="contractions_not_increased", value=True),
        ClinicalFact(code="membranes_intact", value=True),
        ClinicalFact(code="no_cervical_progress", value=True),
        salience=80,
    )
    def l32_latent_discharge_with_return_advice(self, duration, hours):
        self._follow3_emit(
            rule_id="L32", severity="low", category="labour_progress",
            rule_type="Counseling", priority="Medium", confidence="High",
            title="الخروج بعد مرحلة كامنة غير متقدمة",
            detail=(
                "يمكن خروج المرأة مع تنبيه صريح بالعودة فوراً عند الألم أو النزيف "
                "أو تمزق الأغشية."
            ),
            source="كتاب1 D7",
            evidence_codes=("labour_stage_latent", "latent_stage_duration_hours", "contractions_not_increased", "membranes_intact", "no_cervical_progress"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_active", value=True),
        salience=67,
    )
    def l33_active_stage_monitoring(self):
        self._follow3_emit(
            rule_id="L33", severity="low", category="labour_monitoring",
            rule_type="Monitoring", priority="High", confidence="High",
            title="مراقبة المرحلة النشطة",
            detail=(
                "راقب علامات الطوارئ والانقباضات ونبض الجنين والمزاج كل 30 دقيقة، "
                "واتساع عنق الرحم والحرارة والنبض والضغط كل أربع ساعات."
            ),
            source="كتاب1 D8", evidence_codes=("labour_stage_active",),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_active", value=True),
        ClinicalFact(code="active_stage_prolonged", value=True),
        salience=92,
    )
    def l34_active_stage_duration_exceeded(self):
        self._follow3_emit(
            rule_id="L34", severity="medium", category="labour_progress",
            rule_type="Assessment", priority="Medium", confidence="Medium",
            title="تجاوز المدة الطبيعية المتوقعة للمرحلة النشطة",
            detail=(
                "اعتبر تجاوز 12 ساعة في الولادة الأولى أو 10 ساعات في الولادات "
                "اللاحقة سبباً لتقييم سريري شامل، وليس لتحويل آلي فوري."
            ),
            source="كتاب3 rec.6",
            evidence_codes=("active_stage_duration_hours", "primipara", "active_stage_prolonged"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="labour_stage_active", value=True),
        AS.rate << ClinicalFact(code="cervical_dilation_rate_cm_per_hour", value=MATCH.value),
        TEST(lambda value: float(value) < 1.0),
        salience=94,
    )
    def l35_slow_dilation_not_automatic_intervention(self, rate, value):
        self._follow3_emit(
            rule_id="L35", severity="medium", category="labour_progress",
            rule_type="Assessment", priority="High", confidence="High",
            title="بطء الاتساع لا يكفي وحده للتدخل",
            detail=(
                "لا تعتمد معدل أقل من 1 سم/ساعة وحده للتحويل أو التدخل؛ قيّم "
                "الانقباضات وحالة الجنين والحالة العامة للأم قبل القرار."
            ),
            source="كتاب3 rec.7-9",
            evidence_codes=("labour_stage_active", "cervical_dilation_rate_cm_per_hour"),
        )

    # ==================================================================
    # 5. Avoid unjustified acceleration interventions — L36..L41
    # ==================================================================
    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="normal_labour_progress", value=True),
        ClinicalFact(code="no_true_pathological_delay", value=True),
        salience=42,
    )
    def l36_no_routine_amniotomy(self):
        self._follow3_emit(
            rule_id="L36", severity="low", category="avoid_labour_acceleration",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم التوصية ببضع السلى الروتيني",
            detail="لا تُوصِ ببضع السلى الروتيني وحده كإجراء وقائي في مخاض طبيعي التقدم.",
            source="كتاب3 rec.28",
            evidence_codes=("normal_labour_progress", "no_true_pathological_delay"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="normal_labour_progress", value=True),
        ClinicalFact(code="no_true_pathological_delay", value=True),
        salience=41,
    )
    def l37_no_early_amniotomy_oxytocin_package(self):
        self._follow3_emit(
            rule_id="L37", severity="low", category="avoid_labour_acceleration",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم التوصية ببضع السلى والأوكسيتوسين المبكر روتينياً",
            detail="لا تُوصِ ببضع سلى مبكر مقترن بأوكسيتوسين مبكر كإجراء وقائي شامل.",
            source="كتاب3 rec.29",
            evidence_codes=("normal_labour_progress", "no_true_pathological_delay"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="epidural_in_use", value=True),
        ClinicalFact(code="no_true_pathological_delay", value=True),
        salience=43,
    )
    def l38_no_oxytocin_for_epidural_prevention(self):
        self._follow3_emit(
            rule_id="L38", severity="low", category="avoid_labour_acceleration",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم إعطاء الأوكسيتوسين وقائياً بسبب التخدير فوق الجافية",
            detail="لا تُوصِ بإعطاء الأوكسيتوسين لمنع تأخر المخاض عند وجود تخدير فوق الجافية بلا تأخر حقيقي.",
            source="كتاب3 rec.30",
            evidence_codes=("epidural_in_use", "no_true_pathological_delay"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="normal_labour_progress", value=True),
        ClinicalFact(code="no_true_pathological_delay", value=True),
        salience=40,
    )
    def l39_no_antispasmodics_for_prevention(self):
        self._follow3_emit(
            rule_id="L39", severity="low", category="avoid_labour_acceleration",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم استخدام مضادات التشنج وقائياً لتسريع المخاض",
            detail="لا تُوصِ بمضادات التشنج لمنع تأخر المخاض في مخاض طبيعي التقدم.",
            source="كتاب3 rec.31",
            evidence_codes=("normal_labour_progress", "no_true_pathological_delay"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="normal_labour_progress", value=True),
        ClinicalFact(code="no_true_pathological_delay", value=True),
        salience=39,
    )
    def l40_no_iv_fluids_to_shorten_labour(self):
        self._follow3_emit(
            rule_id="L40", severity="low", category="avoid_labour_acceleration",
            rule_type="Prevention", priority="Low", confidence="High",
            title="عدم إعطاء سوائل وريدية بهدف تقصير المخاض فقط",
            detail="لا تُوصِ بسوائل وريدية لمجرد تقصير مدة المخاض في حالة طبيعية التقدم.",
            source="كتاب3 rec.32",
            evidence_codes=("normal_labour_progress", "no_true_pathological_delay"),
        )

    @Rule(
        ClinicalFact(code="in_labour", value=True),
        ClinicalFact(code="normal_labour_progress", value=True),
        ClinicalFact(code="no_true_pathological_delay", value=True),
        salience=38,
    )
    def l41_no_full_active_management_package(self):
        self._follow3_emit(
            rule_id="L41", severity="low", category="avoid_labour_acceleration",
            rule_type="Prevention", priority="Low", confidence="Medium",
            title="عدم تطبيق حزمة الإدارة الفعالة الكاملة روتينياً",
            detail=(
                "لا تُوصِ بحزمة الإدارة الفعالة الكاملة للمخاض كإجراء وقائي شامل "
                "لكل النساء ذوات المخاض الطبيعي التقدم."
            ),
            source="استنتاج من كتاب3 rec.28-32",
            evidence_codes=("normal_labour_progress", "no_true_pathological_delay"),
        )
