"""Delivery/birth rules converted from ``kb_rules_delivery.md``.

Source basis:
    WHO PCPNC 2015 + WHO Intrapartum Care 2018, as structured in the supplied
    Markdown file. Where the supplied source identifies a conflict, the 2018
    recommendation is controlling.

These rules provide clinician-review decision support and are not autonomous
medical orders.
"""
from __future__ import annotations

from experta import NOT, Rule

from ..facts import ClinicalFact, RuleState
from ..suggestions import Suggestion

FOLLOW4_RULES_VERSION = "1.0.0"
SOURCE_NAME = "kb_rules_delivery.md — WHO PCPNC 2015 + WHO Intrapartum Care 2018"


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


class Follow4Rules:
    """Sixty-three atomic delivery and immediate postpartum rules D1–D63."""

    def _follow4_evidence(self, codes) -> list[dict]:
        wanted = set(codes)
        evidence: list[dict] = []
        for fact in getattr(self, "facts", {}).values():
            if isinstance(fact, ClinicalFact) and fact.get("code") in wanted:
                evidence.append(_fact_evidence(fact))
        return evidence

    def _follow4_emit(
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
        full_id = f"FOLLOW4-{rule_id}"
        evidence = self._follow4_evidence(evidence_codes)
        default_chain = [
            f"تحققت شروط القاعدة {full_id}",
            f"الفئة: {category}؛ النوع: {rule_type}",
            f"الأولوية: {priority}؛ قوة الدليل: {confidence}",
            "إصدار توصية للمراجعة السريرية",
        ]
        self._emit(
            Suggestion(
                rule_id=full_id,
                rule_version=FOLLOW4_RULES_VERSION,
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

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        salience=80,
    )
    def d1_second_stage(self):
        self._follow4_emit(
            rule_id='D1',
            severity='low',
            category='second_stage',
            rule_type='Assessment',
            priority='Medium',
            confidence='High',
            title='المرحلة الثانية من الولادة',
            detail='اعتبر المرأة في المرحلة الثانية، وتوقّع اكتمالها خلال 3 ساعات في الولادة الأولى أو ساعتين في الولادات اللاحقة.',
            source='كتاب3 rec.33',
            evidence_codes=('delivery_phase_second',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        ClinicalFact(code='epidural_analgesia', value=False),
        salience=80,
    )
    def d2_second_stage(self):
        self._follow4_emit(
            rule_id='D2',
            severity='low',
            category='second_stage',
            rule_type='Counseling',
            priority='Medium',
            confidence='High',
            title='الوضعية والدفع دون تخدير فوق الجافية',
            detail='شجّع وضعية الولادة التي تختارها المرأة، بما فيها الوضعيات المنتصبة، وشجّع الدفع حسب رغبتها الذاتية دون إلحاح.',
            source='كتاب3 rec.34-35',
            evidence_codes=('epidural_analgesia',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        ClinicalFact(code='epidural_analgesia', value=True),
        salience=80,
    )
    def d3_second_stage(self):
        self._follow4_emit(
            rule_id='D3',
            severity='low',
            category='second_stage',
            rule_type='Counseling',
            priority='Medium',
            confidence='Medium',
            title='الوضعية والدفع مع تخدير فوق الجافية',
            detail='شجّع وضعية الولادة حسب اختيار المرأة، واعرض تأخير الدفع 1-2 ساعة فقط عند توفر مراقبة دقيقة لنقص الأكسجين الجنيني.',
            source='كتاب3 rec.36-37',
            evidence_codes=('epidural_analgesia',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        ClinicalFact(code='fetal_head_visible', value=True),
        salience=118,
    )
    def d4_second_stage(self):
        self._follow4_emit(
            rule_id='D4',
            severity='low',
            category='second_stage',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='الوقاية من تمزق العجان عند ظهور الرأس',
            detail='طبّق تدليك العجان أو الكمادات الدافئة أو الدعم اليدوي حسب التفضيل، وتحكّم تدريجياً بولادة الرأس، واطلب من الأم عدم الدفع والتنفس بثبات أثناء خروج الرأس.',
            source='كتاب3 rec.38 / كتاب1 D9-D10',
            evidence_codes=('fetal_head_visible',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        ClinicalFact(code='episiotomy_indication_present', value=False),
        salience=80,
    )
    def d5_second_stage(self):
        self._follow4_emit(
            rule_id='D5',
            severity='low',
            category='second_stage',
            rule_type='Prevention',
            priority='Low',
            confidence='High',
            title='عدم إجراء بضع العجان روتينياً',
            detail='لا تُوصِ ببضع العجان الروتيني أو المتحرر عند عدم وجود انسداد واضح أو مجيء مقعدي أو عسر كتفين.',
            source='كتاب3 rec.39',
            evidence_codes=('episiotomy_indication_present',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        ClinicalFact(code='episiotomy_indication_present', value=True),
        salience=125,
    )
    def d6_second_stage(self):
        self._follow4_emit(
            rule_id='D6',
            severity='medium',
            category='second_stage',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='بضع عجان عند وجود مؤشر واضح',
            detail='أجرِ بضع عجان كافياً عند وجود انسداد بسبب ثآليل أو ندبات أو تمزق درجة ثالثة سابق، أو عند المجيء المقعدي أو عسر الكتفين.',
            source='كتاب1 D9',
            evidence_codes=('episiotomy_indication_present',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        salience=80,
    )
    def d7_second_stage(self):
        self._follow4_emit(
            rule_id='D7',
            severity='low',
            category='second_stage',
            rule_type='Prevention',
            priority='Low',
            confidence='High',
            title='منع الضغط على قاع الرحم',
            detail='لا تُوصِ بتطبيق ضغط يدوي على قاع الرحم لتسهيل الولادة.',
            source='كتاب3 rec.40',
            evidence_codes=('delivery_phase_second',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_second', value=True),
        ClinicalFact(code='second_stage_duration_ge_2h', value=True),
        ClinicalFact(code='visible_head_descent', value=False),
        salience=135,
    )
    def d8_second_stage(self):
        self._follow4_emit(
            rule_id='D8',
            severity='high',
            category='second_stage',
            rule_type='Referral',
            priority='High',
            confidence='High',
            title='تأخر هبوط رأس الجنين في المرحلة الثانية',
            detail='استدعِ طاقماً مدرّباً على الشفط المهبلي أو حوّل عاجلاً للمستشفى عند مرور ساعتين أو أكثر دون هبوط ثابت ومرئي للرأس.',
            source='كتاب1 D10',
            evidence_codes=('second_stage_duration_ge_2h', 'visible_head_descent'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='fetal_head_born', value=True),
        salience=128,
    )
    def d9_second_stage(self):
        self._follow4_emit(
            rule_id='D9',
            severity='low',
            category='second_stage',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='فحص الحبل السري حول الرقبة',
            detail='بعد ولادة الرأس افحص وجود الحبل حول الرقبة؛ أزحه إن كان رخواً، أو اربطه واقطعه إن كان مشدوداً.',
            source='كتاب1 D10',
            evidence_codes=('fetal_head_born',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='fetal_head_born', value=True),
        ClinicalFact(code='cord_checked', value=True),
        RuleState(code='FOLLOW4-D9'),
        salience=116,
    )
    def d10_second_stage(self):
        self._follow4_emit(
            rule_id='D10',
            severity='low',
            category='second_stage',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='انتظار دوران الأكتاف التلقائي',
            detail='بعد ولادة الرأس وفحص الحبل، انتظر دوران الأكتاف التلقائي لمدة 1-2 دقيقة قبل المساعدة اليدوية.',
            source='كتاب1 D10',
            evidence_codes=('fetal_head_born', 'cord_checked'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='newborn_fully_born', value=True),
        salience=130,
    )
    def d11_second_stage(self):
        self._follow4_emit(
            rule_id='D11',
            severity='low',
            category='second_stage',
            rule_type='Assessment',
            priority='High',
            confidence='High',
            title='تجفيف المولود وتقييم التنفس',
            detail='جفّف المولود فوراً وقيّم تنفسه في اللحظة نفسها أثناء التجفيف.',
            source='كتاب1 D10',
            evidence_codes=('newborn_fully_born',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='newborn_fully_born', value=True),
        ClinicalFact(code='newborn_not_breathing_or_gasping', value=True),
        salience=180,
    )
    def d12_newborn_emergency(self):
        self._follow4_emit(
            rule_id='D12',
            severity='critical',
            category='newborn_emergency',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='بدء إنعاش المولود فوراً',
            detail='إذا لم يتنفس المولود أو كان يلهث بعد التجفيف، اقطع الحبل بسرعة وابدأ إنعاش المولود فوراً؛ هذه القاعدة تتجاوز تأخير قطع الحبل في D28.',
            source='كتاب1 D10',
            evidence_codes=('newborn_not_breathing_or_gasping',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='first_newborn_born', value=True),
        salience=145,
    )
    def d13_second_stage(self):
        self._follow4_emit(
            rule_id='D13',
            severity='low',
            category='second_stage',
            rule_type='Assessment',
            priority='High',
            confidence='High',
            title='استبعاد وجود مولود ثانٍ',
            detail='بعد ولادة المولود الأول، استبعد وجود مولود ثانٍ قبل إعطاء الأوكسيتوسين للأم.',
            source='كتاب1 D10',
            evidence_codes=('first_newborn_born',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='cord_prolapse_visible_or_palpable', value=True),
        ClinicalFact(code='prolapsed_cord_pulsating', value=True),
        salience=175,
    )
    def d14_delivery_emergency(self):
        self._follow4_emit(
            rule_id='D14',
            severity='critical',
            category='delivery_emergency',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='تدبير تدلي الحبل السري النابض',
            detail='ضع الأم بوضعية الركبة-الصدر أو ارفع الوركين لتخفيف الضغط، وشجّع الدفع مع الانقباضات لتسريع الولادة إذا كان الاتساع كاملاً.',
            source='كتاب1 D14',
            evidence_codes=('cord_prolapse_visible_or_palpable', 'prolapsed_cord_pulsating'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='cord_prolapse_visible_or_palpable', value=True),
        ClinicalFact(code='prolapsed_cord_pulsating', value=False),
        salience=174,
    )
    def d15_delivery_emergency(self):
        self._follow4_emit(
            rule_id='D15',
            severity='critical',
            category='delivery_emergency',
            rule_type='Counseling',
            priority='High',
            confidence='High',
            title='تدلي حبل سري غير نابض',
            detail='أبلغ الأهل بحساسية أن الجنين على الأرجح متوفى عندما يكون الحبل السري المتدلي غير نابض.',
            source='كتاب1 D14',
            evidence_codes=('cord_prolapse_visible_or_palpable', 'prolapsed_cord_pulsating'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='breech_presentation', value=True),
        ClinicalFact(code='birth_imminent', value=False),
        salience=150,
    )
    def d16_delivery_emergency(self):
        self._follow4_emit(
            rule_id='D16',
            severity='high',
            category='delivery_emergency',
            rule_type='Referral',
            priority='High',
            confidence='High',
            title='المجيء المقعدي مع مخاض مبكر',
            detail='حوّل عاجلاً للمستشفى عند ظهور مجيء مقعدي والمخاض ما زال مبكراً.',
            source='كتاب1 D15',
            evidence_codes=('breech_presentation', 'birth_imminent'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='breech_presentation', value=True),
        ClinicalFact(code='birth_imminent', value=True),
        salience=155,
    )
    def d17_delivery_emergency(self):
        self._follow4_emit(
            rule_id='D17',
            severity='high',
            category='delivery_emergency',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='المجيء المقعدي والولادة وشيكة',
            detail='استدعِ مساعدة إضافية، وأكّد الاتساع الكامل، وجهّز إنعاش المولود، وامنع الدفع قبل اكتمال الاتساع وامنع الجذب على جسم الجنين.',
            source='كتاب1 D15',
            evidence_codes=('breech_presentation', 'birth_imminent'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='fetal_head_born', value=True),
        ClinicalFact(code='shoulder_dystocia', value=True),
        salience=178,
    )
    def d18_delivery_emergency(self):
        self._follow4_emit(
            rule_id='D18',
            severity='critical',
            category='delivery_emergency',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='عسر الكتفين',
            detail='طبّق مناورة مكروبرتس وبضع عجان كافياً وضغطاً فوق العانة، وتجنّب الجذب المفرط على الرأس، وحوّل عاجلاً إن فشلت المحاولات.',
            source='كتاب1 D16',
            evidence_codes=('fetal_head_born', 'shoulder_dystocia'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='multiple_pregnancy', value=True),
        salience=150,
    )
    def d19_delivery_emergency(self):
        self._follow4_emit(
            rule_id='D19',
            severity='high',
            category='delivery_emergency',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='الولادة المتعددة',
            detail='جهّز أدوات مضاعفة، ولا تعطِ الأوكسيتوسين ولا تحاول ولادة المشيمة حتى يولد جميع الأطفال، وراقب النزيف بعد الولادة بشكل مكثف.',
            source='كتاب1 D17',
            evidence_codes=('multiple_pregnancy',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='newborn_fully_born', value=True),
        ClinicalFact(code='additional_fetus_excluded', value=True),
        salience=125,
    )
    def d20_third_stage(self):
        self._follow4_emit(
            rule_id='D20',
            severity='low',
            category='third_stage',
            rule_type='Medication',
            priority='High',
            confidence='High',
            title='الأوكسيتوسين الوقائي بعد الولادة',
            detail='بعد ولادة المولود واستبعاد وجود مولود إضافي، أعطِ أوكسيتوسين 10 وحدات دولية عضلياً أو وريدياً للوقاية من نزيف ما بعد الولادة، أو استخدم البديل المتاح.',
            source='كتاب3 rec.41-43',
            evidence_codes=('newborn_fully_born', 'additional_fetus_excluded'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='skilled_birth_attendant_available', value=True),
        ClinicalFact(code='vaginal_birth', value=True),
        salience=80,
    )
    def d21_third_stage(self):
        self._follow4_emit(
            rule_id='D21',
            severity='low',
            category='third_stage',
            rule_type='Treatment',
            priority='Medium',
            confidence='High',
            title='شد الحبل المتحكم',
            detail='عند توفر كادر ماهر وفي ولادة مهبلية، طبّق شد الحبل المتحكم مع ثبات مضاد فوق العانة أثناء انقباض قوي فقط.',
            source='كتاب3 rec.45 / كتاب1 D11',
            evidence_codes=('skilled_birth_attendant_available', 'vaginal_birth'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='placenta_not_delivered_30min_after_oxytocin', value=True),
        ClinicalFact(code='active_bleeding', value=False),
        salience=80,
    )
    def d22_third_stage(self):
        self._follow4_emit(
            rule_id='D22',
            severity='low',
            category='third_stage',
            rule_type='Treatment',
            priority='Medium',
            confidence='High',
            title='المشيمة لم تولد بعد 30 دقيقة',
            detail='فرّغ المثانة، وشجّع الرضاعة، وأعد محاولة شد الحبل المتحكم عند مرور 30 دقيقة بعد الأوكسيتوسين دون نزيف أو ولادة للمشيمة.',
            source='كتاب1 D11',
            evidence_codes=('placenta_not_delivered_30min_after_oxytocin', 'active_bleeding'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='bleeding_while_awaiting_placenta', value=True),
        salience=180,
    )
    def d23_third_stage(self):
        self._follow4_emit(
            rule_id='D23',
            severity='critical',
            category='third_stage',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='نزيف أثناء انتظار المشيمة',
            detail='أدِر الحالة كنزيف حاد فوراً وبأولوية مطلقة، ولا تنتظر ولادة المشيمة.',
            source='كتاب1 D11-D12',
            evidence_codes=('bleeding_while_awaiting_placenta',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='placenta_not_delivered_60min', value=True),
        salience=145,
    )
    def d24_third_stage(self):
        self._follow4_emit(
            rule_id='D24',
            severity='high',
            category='third_stage',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='احتباس المشيمة لمدة 60 دقيقة',
            detail='أزل المشيمة يدوياً وأعطِ مضاداً حيوياً وقائياً عند مرور 60 دقيقة دون ولادتها.',
            source='كتاب1 D12',
            evidence_codes=('placenta_not_delivered_60min',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='manual_placenta_removal_failed', value=True),
        salience=175,
    )
    def d25_third_stage(self):
        self._follow4_emit(
            rule_id='D25',
            severity='critical',
            category='third_stage',
            rule_type='Referral',
            priority='High',
            confidence='High',
            title='تعذّر الإزالة اليدوية للمشيمة',
            detail='حوّل عاجلاً مع خط وريدي وأوكسيتوسين مستمر أثناء النقل إذا تعذرت الإزالة اليدوية للمشيمة.',
            source='كتاب1 D12',
            evidence_codes=('manual_placenta_removal_failed',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='placenta_incomplete', value=True),
        salience=115,
    )
    def d26_third_stage(self):
        self._follow4_emit(
            rule_id='D26',
            severity='medium',
            category='third_stage',
            rule_type='Treatment',
            priority='Medium',
            confidence='High',
            title='المشيمة غير مكتملة',
            detail='أزل الأجزاء المتبقية يدوياً وأعطِ مضاداً حيوياً عندما تكون المشيمة غير مكتملة بعد الفحص.',
            source='كتاب1 D12',
            evidence_codes=('placenta_incomplete',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_third', value=True),
        ClinicalFact(code='prophylactic_oxytocin_given', value=True),
        salience=80,
    )
    def d27_third_stage(self):
        self._follow4_emit(
            rule_id='D27',
            severity='low',
            category='third_stage',
            rule_type='Prevention',
            priority='Low',
            confidence='High',
            title='عدم التدليك المستمر للرحم بعد الأوكسيتوسين',
            detail='لا تُوصِ بالتدليك المستمر الإضافي للرحم كوقاية عندما أُعطي الأوكسيتوسين الوقائي بالفعل.',
            source='كتاب3 rec.46',
            evidence_codes=('prophylactic_oxytocin_given',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='newborn_fully_born', value=True),
        ClinicalFact(code='newborn_resuscitation_needed', value=False),
        NOT(RuleState(code='FOLLOW4-D12')),
        salience=80,
    )
    def d28_third_stage(self):
        self._follow4_emit(
            rule_id='D28',
            severity='low',
            category='third_stage',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='تأخير قطع الحبل السري',
            detail='أخّر قطع الحبل السري لمدة لا تقل عن دقيقة واحدة، ويفضل 1-3 دقائق، إذا لم توجد حاجة طارئة فورية للإنعاش.',
            source='كتاب3 rec.44 / كتاب1 D10',
            evidence_codes=('newborn_fully_born', 'newborn_resuscitation_needed'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_newborn', value=True),
        ClinicalFact(code='amniotic_fluid_clear', value=True),
        ClinicalFact(code='newborn_breathing_spontaneously', value=True),
        salience=80,
    )
    def d29_newborn_care(self):
        self._follow4_emit(
            rule_id='D29',
            severity='low',
            category='newborn_care',
            rule_type='Prevention',
            priority='Low',
            confidence='High',
            title='عدم الشفط الروتيني للفم والأنف',
            detail='لا تُوصِ بالشفط الروتيني للفم والأنف عندما يكون السائل الأمنيوسي صافياً ويبدأ المولود التنفس تلقائياً.',
            source='كتاب3 rec.47',
            evidence_codes=('amniotic_fluid_clear', 'newborn_breathing_spontaneously'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_newborn', value=True),
        ClinicalFact(code='uncomplicated_newborn', value=True),
        salience=120,
    )
    def d30_newborn_care(self):
        self._follow4_emit(
            rule_id='D30',
            severity='low',
            category='newborn_care',
            rule_type='Treatment',
            priority='High',
            confidence='High',
            title='التماس الجلدي المباشر',
            detail='حافظ على تماس جلدي مباشر بين المولود غير المعقد والأم خلال الساعة الأولى للوقاية من انخفاض الحرارة وتعزيز الرضاعة.',
            source='كتاب3 rec.48',
            evidence_codes=('uncomplicated_newborn',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_newborn', value=True),
        ClinicalFact(code='newborn_clinically_stable', value=True),
        ClinicalFact(code='mother_ready_to_breastfeed', value=True),
        salience=118,
    )
    def d31_newborn_care(self):
        self._follow4_emit(
            rule_id='D31',
            severity='low',
            category='newborn_care',
            rule_type='Counseling',
            priority='High',
            confidence='High',
            title='بدء الرضاعة في أقرب وقت',
            detail='ضع المولود المستقر على الثدي في أقرب وقت ممكن عندما تكون الأم جاهزة، ويشمل ذلك منخفضي الوزن القادرين على الرضاعة.',
            source='كتاب3 rec.49',
            evidence_codes=('newborn_clinically_stable', 'mother_ready_to_breastfeed'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_newborn', value=True),
        ClinicalFact(code='first_hour_after_birth_elapsed', value=True),
        ClinicalFact(code='skin_to_skin_completed', value=True),
        ClinicalFact(code='breastfeeding_started', value=True),
        salience=115,
    )
    def d32_newborn_care(self):
        self._follow4_emit(
            rule_id='D32',
            severity='low',
            category='newborn_care',
            rule_type='Medication',
            priority='High',
            confidence='High',
            title='إعطاء فيتامين K',
            detail='بعد الساعة الأولى وإتمام التماس الجلدي وبدء الرضاعة، أعطِ المولود 1 ملغ فيتامين K عضلياً.',
            source='كتاب3 rec.50',
            evidence_codes=('first_hour_after_birth_elapsed', 'skin_to_skin_completed', 'breastfeeding_started'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_newborn', value=True),
        ClinicalFact(code='newborn_fully_born', value=True),
        salience=80,
    )
    def d33_newborn_care(self):
        self._follow4_emit(
            rule_id='D33',
            severity='low',
            category='newborn_care',
            rule_type='Prevention',
            priority='Medium',
            confidence='High',
            title='تأخير استحمام المولود وعدم فصله',
            detail='أخّر الاستحمام حتى 24 ساعة، أو 6 ساعات كحد أدنى، وأبقِ المولود في غرفة الأم طوال اليوم دون فصل.',
            source='كتاب3 rec.51',
            evidence_codes=('newborn_fully_born',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='delivery_completed', value=True),
        salience=120,
    )
    def d34_postpartum_assessment(self):
        self._follow4_emit(
            rule_id='D34',
            severity='low',
            category='postpartum_assessment',
            rule_type='Assessment',
            priority='High',
            confidence='High',
            title='فحص توتر الرحم بعد الولادة',
            detail='افحص توتر الرحم بطريق البطن بعد انتهاء الولادة للكشف المبكر عن ارتخاء الرحم.',
            source='كتاب3 rec.52',
            evidence_codes=('delivery_completed',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='postpartum_first_hour', value=True),
        salience=115,
    )
    def d35_postpartum_monitoring(self):
        self._follow4_emit(
            rule_id='D35',
            severity='low',
            category='postpartum_monitoring',
            rule_type='Monitoring',
            priority='High',
            confidence='High',
            title='مراقبة الساعة الأولى بعد الولادة',
            detail='راقب النزيف وانقباض الرحم والتنفس والدفء كل 15 دقيقة خلال الساعة الأولى بعد الولادة.',
            source='كتاب3 rec.55 / كتاب1 D18',
            evidence_codes=('postpartum_first_hour',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='postpartum_hours_2_to_4', value=True),
        salience=115,
    )
    def d36_postpartum_monitoring(self):
        self._follow4_emit(
            rule_id='D36',
            severity='low',
            category='postpartum_monitoring',
            rule_type='Monitoring',
            priority='High',
            confidence='High',
            title='مراقبة الساعات 2-4 بعد الولادة',
            detail='راقب النزيف وانقباض الرحم والحرارة والنبض كل ساعة خلال الساعات الثانية إلى الرابعة بعد الولادة.',
            source='كتاب3 rec.55 / كتاب1 D19',
            evidence_codes=('postpartum_hours_2_to_4',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='postpartum_hours_4_to_24', value=True),
        salience=115,
    )
    def d37_postpartum_monitoring(self):
        self._follow4_emit(
            rule_id='D37',
            severity='low',
            category='postpartum_monitoring',
            rule_type='Monitoring',
            priority='High',
            confidence='High',
            title='المراقبة من 4 إلى 24 ساعة',
            detail='راقب كل أربع ساعات بعد مرور 4 ساعات وحتى 24 ساعة، ووثّق حدوث التبول خلال 6 ساعات من الولادة.',
            source='كتاب1 D19-D20',
            evidence_codes=('postpartum_hours_4_to_24',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='pad_saturated_under_5min', value=True),
        salience=185,
    )
    def d38_postpartum_emergency_v1(self):
        self._follow4_emit(
            rule_id='D38',
            severity='critical',
            category='postpartum_emergency',
            rule_type='Diagnosis',
            priority='High',
            confidence='High',
            title='نزيف حاد بعد الولادة',
            detail='شخّص نزيفاً حاداً، وأدِره فوراً، وحوّل عاجلاً عند التشبع السريع للفوط أو عدم تماسك الرحم.',
            source='كتاب1 D20',
            evidence_codes=('pad_saturated_under_5min', 'more_than_one_pad_in_5min', 'uterus_not_firm'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='more_than_one_pad_in_5min', value=True),
        salience=185,
    )
    def d38_postpartum_emergency_v2(self):
        self._follow4_emit(
            rule_id='D38',
            severity='critical',
            category='postpartum_emergency',
            rule_type='Diagnosis',
            priority='High',
            confidence='High',
            title='نزيف حاد بعد الولادة',
            detail='شخّص نزيفاً حاداً، وأدِره فوراً، وحوّل عاجلاً عند التشبع السريع للفوط أو عدم تماسك الرحم.',
            source='كتاب1 D20',
            evidence_codes=('pad_saturated_under_5min', 'more_than_one_pad_in_5min', 'uterus_not_firm'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='uterus_not_firm', value=True),
        salience=185,
    )
    def d38_postpartum_emergency_v3(self):
        self._follow4_emit(
            rule_id='D38',
            severity='critical',
            category='postpartum_emergency',
            rule_type='Diagnosis',
            priority='High',
            confidence='High',
            title='نزيف حاد بعد الولادة',
            detail='شخّص نزيفاً حاداً، وأدِره فوراً، وحوّل عاجلاً عند التشبع السريع للفوط أو عدم تماسك الرحم.',
            source='كتاب1 D20',
            evidence_codes=('pad_saturated_under_5min', 'more_than_one_pad_in_5min', 'uterus_not_firm'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='diastolic_bp_ge_110', value=True),
        salience=182,
    )
    def d39_postpartum_hypertension_v1(self):
        self._follow4_emit(
            rule_id='D39',
            severity='critical',
            category='postpartum_hypertension',
            rule_type='Diagnosis',
            priority='High',
            confidence='High',
            title='تسمم حمل شديد بعد الولادة',
            detail='شخّص تسمم حمل شديداً، وأعطِ كبريتات المغنيزيوم، ولا تعطِ الإرغومترين، وحوّل عاجلاً.',
            source='كتاب1 D21',
            evidence_codes=('diastolic_bp_ge_110', 'diastolic_bp_ge_90', 'bp_readings_at_least_2', 'proteinuria_ge_2', 'severe_preeclampsia_symptom'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='diastolic_bp_ge_90', value=True),
        ClinicalFact(code='bp_readings_at_least_2', value=True),
        ClinicalFact(code='proteinuria_ge_2', value=True),
        ClinicalFact(code='severe_preeclampsia_symptom', value=True),
        salience=182,
    )
    def d39_postpartum_hypertension_v2(self):
        self._follow4_emit(
            rule_id='D39',
            severity='critical',
            category='postpartum_hypertension',
            rule_type='Diagnosis',
            priority='High',
            confidence='High',
            title='تسمم حمل شديد بعد الولادة',
            detail='شخّص تسمم حمل شديداً، وأعطِ كبريتات المغنيزيوم، ولا تعطِ الإرغومترين، وحوّل عاجلاً.',
            source='كتاب1 D21',
            evidence_codes=('diastolic_bp_ge_110', 'diastolic_bp_ge_90', 'bp_readings_at_least_2', 'proteinuria_ge_2', 'severe_preeclampsia_symptom'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='diastolic_bp_90_to_110', value=True),
        ClinicalFact(code='bp_readings_at_least_2', value=True),
        ClinicalFact(code='proteinuria_ge_2', value=True),
        ClinicalFact(code='severe_preeclampsia_symptom', value=False),
        NOT(RuleState(code='FOLLOW4-D39')),
        salience=150,
    )
    def d40_postpartum_hypertension(self):
        self._follow4_emit(
            rule_id='D40',
            severity='high',
            category='postpartum_hypertension',
            rule_type='Diagnosis',
            priority='High',
            confidence='High',
            title='تسمم حمل بعد الولادة',
            detail='شخّص تسمم الحمل، وراقب الضغط كل ساعة، ولا تعطِ الإرغومترين، وحوّل إن استمر الارتفاع.',
            source='كتاب1 D21',
            evidence_codes=('diastolic_bp_90_to_110', 'bp_readings_at_least_2', 'proteinuria_ge_2', 'severe_preeclampsia_symptom'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='diastolic_bp_ge_90', value=True),
        ClinicalFact(code='bp_readings_at_least_2', value=True),
        ClinicalFact(code='proteinuria_absent', value=True),
        NOT(RuleState(code='FOLLOW4-D39')),
        salience=125,
    )
    def d41_postpartum_hypertension(self):
        self._follow4_emit(
            rule_id='D41',
            severity='medium',
            category='postpartum_hypertension',
            rule_type='Diagnosis',
            priority='Medium',
            confidence='High',
            title='ارتفاع ضغط الدم بعد الولادة',
            detail='شخّص ارتفاع ضغط الدم، وراقب كل ساعة، ولا تعطِ الإرغومترين عندما يكون الضغط مرتفعاً على قراءتين دون بروتينية.',
            source='كتاب1 D21',
            evidence_codes=('diastolic_bp_ge_90', 'bp_readings_at_least_2', 'proteinuria_absent'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='hemoglobin_lt_7', value=True),
        salience=150,
    )
    def d42_postpartum_anemia_v1(self):
        self._follow4_emit(
            rule_id='D42',
            severity='high',
            category='postpartum_anemia',
            rule_type='Referral',
            priority='High',
            confidence='High',
            title='فقر دم شديد بعد الولادة',
            detail='شخّص فقر دم شديداً وحوّل عاجلاً عند هيموغلوبين أقل من 7 غ/دل أو شحوب شديد.',
            source='كتاب1 D22',
            evidence_codes=('hemoglobin_lt_7', 'severe_pallor'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='severe_pallor', value=True),
        salience=150,
    )
    def d42_postpartum_anemia_v2(self):
        self._follow4_emit(
            rule_id='D42',
            severity='high',
            category='postpartum_anemia',
            rule_type='Referral',
            priority='High',
            confidence='High',
            title='فقر دم شديد بعد الولادة',
            detail='شخّص فقر دم شديداً وحوّل عاجلاً عند هيموغلوبين أقل من 7 غ/دل أو شحوب شديد.',
            source='كتاب1 D22',
            evidence_codes=('hemoglobin_lt_7', 'severe_pallor'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='hemoglobin_7_to_11', value=True),
        salience=120,
    )
    def d43_postpartum_anemia_v1(self):
        self._follow4_emit(
            rule_id='D43',
            severity='medium',
            category='postpartum_anemia',
            rule_type='Medication',
            priority='Medium',
            confidence='High',
            title='فقر دم متوسط بعد الولادة',
            detail='شخّص فقر دم متوسطاً، وأعطِ حديداً مضاعفاً، وحدد متابعة بعد 3 أيام ثم 4 أسابيع.',
            source='كتاب1 D22',
            evidence_codes=('hemoglobin_7_to_11', 'mild_pallor'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='mild_pallor', value=True),
        salience=120,
    )
    def d43_postpartum_anemia_v2(self):
        self._follow4_emit(
            rule_id='D43',
            severity='medium',
            category='postpartum_anemia',
            rule_type='Medication',
            priority='Medium',
            confidence='High',
            title='فقر دم متوسط بعد الولادة',
            detail='شخّص فقر دم متوسطاً، وأعطِ حديداً مضاعفاً، وحدد متابعة بعد 3 أيام ثم 4 أسابيع.',
            source='كتاب1 D22',
            evidence_codes=('hemoglobin_7_to_11', 'mild_pallor'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='hemoglobin_gt_11', value=True),
        salience=80,
    )
    def d44_postpartum_anemia(self):
        self._follow4_emit(
            rule_id='D44',
            severity='low',
            category='postpartum_anemia',
            rule_type='Prevention',
            priority='Low',
            confidence='High',
            title='لا يوجد فقر دم بعد الولادة',
            detail='عندما يكون الهيموغلوبين أكبر من 11 غ/دل، أعطِ الحديد الوقائي لمدة ثلاثة أشهر.',
            source='كتاب1 D22',
            evidence_codes=('hemoglobin_gt_11',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='temperature_gt_38', value=True),
        ClinicalFact(code='postpartum_infection_additional_sign', value=True),
        salience=180,
    )
    def d45_postpartum_infection(self):
        self._follow4_emit(
            rule_id='D45',
            severity='critical',
            category='postpartum_infection',
            rule_type='Diagnosis',
            priority='High',
            confidence='High',
            title='التهاب رحمي وجنيني بعد الولادة',
            detail='شخّص التهاباً رحمياً وجنينياً، وأعطِ خطاً وريدياً وسوائل ومضاداً حيوياً وأوكسيتوسين إن وُلدت المشيمة، وحوّل عاجلاً.',
            source='كتاب1 D23',
            evidence_codes=('temperature_gt_38', 'postpartum_infection_additional_sign'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='persistent_temperature_gt_38', value=True),
        ClinicalFact(code='postpartum_infection_additional_sign', value=False),
        salience=120,
    )
    def d46_postpartum_infection(self):
        self._follow4_emit(
            rule_id='D46',
            severity='medium',
            category='postpartum_infection',
            rule_type='Diagnosis',
            priority='Medium',
            confidence='High',
            title='خطر عدوى مع استمرار الحرارة',
            detail='أعطِ سوائل فموية، وراقب الحرارة كل أربع ساعات، وأعطِ مضاداً حيوياً وحوّل إذا استمرت الحرارة أكثر من 12 ساعة دون علامات D45.',
            source='كتاب1 D23',
            evidence_codes=('persistent_temperature_gt_38', 'postpartum_infection_additional_sign'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='perineal_tear_extends_to_anus_or_rectum', value=True),
        salience=150,
    )
    def d47_perineal_trauma(self):
        self._follow4_emit(
            rule_id='D47',
            severity='high',
            category='perineal_trauma',
            rule_type='Referral',
            priority='High',
            confidence='High',
            title='تمزق عجان من الدرجة الثالثة',
            detail='شخّص تمزقاً يمتد إلى الشرج أو المستقيم وحوّل عاجلاً للمستشفى.',
            source='كتاب1 D24',
            evidence_codes=('perineal_tear_extends_to_anus_or_rectum',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='simple_perineal_tear_or_episiotomy', value=True),
        ClinicalFact(code='ongoing_bleeding', value=True),
        salience=120,
    )
    def d48_perineal_trauma(self):
        self._follow4_emit(
            rule_id='D48',
            severity='medium',
            category='perineal_trauma',
            rule_type='Treatment',
            priority='Medium',
            confidence='High',
            title='خياطة تمزق العجان أو بضع الفرج',
            detail='اخِط التمزق البسيط أو موضع بضع الفرج عند وجود نزيف مستمر.',
            source='كتاب1 D24',
            evidence_codes=('simple_perineal_tear_or_episiotomy', 'ongoing_bleeding'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='uncomplicated_vaginal_birth', value=True),
        salience=80,
    )
    def d49_antibiotic_stewardship(self):
        self._follow4_emit(
            rule_id='D49',
            severity='low',
            category='antibiotic_stewardship',
            rule_type='Prevention',
            priority='Low',
            confidence='High',
            title='عدم المضاد الوقائي بعد ولادة مهبلية غير معقدة',
            detail='لا تُوصِ بمضاد حيوي وقائي روتيني بعد ولادة مهبلية غير معقدة.',
            source='كتاب3 rec.53',
            evidence_codes=('uncomplicated_vaginal_birth',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_postpartum', value=True),
        ClinicalFact(code='uncomplicated_episiotomy', value=True),
        salience=80,
    )
    def d50_antibiotic_stewardship(self):
        self._follow4_emit(
            rule_id='D50',
            severity='low',
            category='antibiotic_stewardship',
            rule_type='Prevention',
            priority='Low',
            confidence='High',
            title='عدم المضاد الوقائي لبضع فرج غير معقد',
            detail='لا تُوصِ بمضاد حيوي وقائي روتيني لبضع الفرج دون مضاعفات.',
            source='كتاب3 rec.54',
            evidence_codes=('uncomplicated_episiotomy',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='rpr_test_done', value=False),
        salience=120,
    )
    def d51_predischarge_prevention(self):
        self._follow4_emit(
            rule_id='D51',
            severity='low',
            category='predischarge_prevention',
            rule_type='Screening',
            priority='High',
            confidence='High',
            title='فحص RPR قبل الخروج',
            detail='أجرِ فحص RPR قبل الخروج إذا لم يُجرَ خلال الحمل.',
            source='كتاب1 D25',
            evidence_codes=('rpr_test_done',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='rpr_positive', value=True),
        salience=145,
    )
    def d52_predischarge_prevention(self):
        self._follow4_emit(
            rule_id='D52',
            severity='high',
            category='predischarge_prevention',
            rule_type='Medication',
            priority='High',
            confidence='High',
            title='علاج RPR الإيجابي',
            detail='أعطِ بنسلين بنزاثين للأم والشريك، وخطط لعلاج المولود عند إيجابية RPR.',
            source='كتاب1 D25',
            evidence_codes=('rpr_positive',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='tetanus_vaccination_incomplete', value=True),
        salience=115,
    )
    def d53_predischarge_prevention(self):
        self._follow4_emit(
            rule_id='D53',
            severity='low',
            category='predischarge_prevention',
            rule_type='Medication',
            priority='High',
            confidence='High',
            title='استكمال تطعيم التيتانوس قبل الخروج',
            detail='أعطِ الجرعة المستحقة من لقاح التيتانوس قبل الخروج عندما لم يكتمل التطعيم.',
            source='كتاب1 D25',
            evidence_codes=('tetanus_vaccination_incomplete',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='before_facility_discharge', value=True),
        salience=80,
    )
    def d54_predischarge_prevention(self):
        self._follow4_emit(
            rule_id='D54',
            severity='low',
            category='predischarge_prevention',
            rule_type='Medication',
            priority='High',
            confidence='High',
            title='الحديد والفوليك أسيد قبل الخروج',
            detail='أعطِ الحديد وحمض الفوليك لمدة ثلاثة أشهر، وشجّع النوم تحت ناموسية مشبعة بمبيد.',
            source='كتاب1 D25',
            evidence_codes=('before_facility_discharge',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='hiv_status_unknown', value=True),
        salience=120,
    )
    def d55_predischarge_prevention(self):
        self._follow4_emit(
            rule_id='D55',
            severity='low',
            category='predischarge_prevention',
            rule_type='Screening',
            priority='High',
            confidence='High',
            title='عرض فحص HIV قبل الخروج',
            detail='اعرض فحص HIV السريع قبل الخروج إذا كانت الحالة غير معروفة.',
            source='كتاب1 D25',
            evidence_codes=('hiv_status_unknown',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='uncomplicated_vaginal_birth', value=True),
        ClinicalFact(code='facility_birth', value=True),
        salience=115,
    )
    def d56_discharge(self):
        self._follow4_emit(
            rule_id='D56',
            severity='low',
            category='discharge',
            rule_type='Assessment',
            priority='High',
            confidence='High',
            title='البقاء 24 ساعة بعد الولادة',
            detail='أبقِ الأم والمولود في المنشأة الصحية مدة 24 ساعة على الأقل بعد ولادة مهبلية غير معقدة.',
            source='كتاب3 rec.56 / كتاب1 D28',
            evidence_codes=('uncomplicated_vaginal_birth', 'facility_birth'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='postpartum_immediate_danger_sign', value=True),
        salience=185,
    )
    def d57_postpartum_danger_signs(self):
        self._follow4_emit(
            rule_id='D57',
            severity='critical',
            category='postpartum_danger_signs',
            rule_type='Referral',
            priority='High',
            confidence='High',
            title='علامة خطر بعد الولادة: مستشفى فوراً',
            detail='وجّه المرأة إلى المستشفى فوراً دون انتظار، ليلاً أو نهاراً، عند وجود علامة خطر حادة بعد الولادة.',
            source='كتاب1 D28',
            evidence_codes=('postpartum_immediate_danger_sign',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='postpartum_early_review_sign', value=True),
        salience=125,
    )
    def d58_postpartum_danger_signs(self):
        self._follow4_emit(
            rule_id='D58',
            severity='medium',
            category='postpartum_danger_signs',
            rule_type='Referral',
            priority='Medium',
            confidence='High',
            title='مراجعة مبكرة بعد الولادة',
            detail='وجّه المرأة للمراجعة بأقرب وقت عند وجود أعراض تستدعي التقييم، مع التعامل الفوري والحساس مع الاكتئاب الشديد أو الأفكار الانتحارية.',
            source='كتاب1 D28',
            evidence_codes=('postpartum_early_review_sign',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='at_facility_discharge', value=True),
        salience=80,
    )
    def d59_discharge(self):
        self._follow4_emit(
            rule_id='D59',
            severity='low',
            category='discharge',
            rule_type='Counseling',
            priority='Medium',
            confidence='High',
            title='جدول متابعة ما بعد الولادة',
            detail='حدد التماس الأول خلال 24 ساعة، والثاني في اليوم 3، والثالث بين اليوم 7-14، والنهائي في الأسبوع السادس.',
            source='كتاب1 D27',
            evidence_codes=('at_facility_discharge',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='before_facility_discharge', value=True),
        salience=80,
    )
    def d60_family_planning(self):
        self._follow4_emit(
            rule_id='D60',
            severity='low',
            category='family_planning',
            rule_type='Counseling',
            priority='Medium',
            confidence='High',
            title='المباعدة بين الحملين',
            detail='استفسر عن الرغبة في إنجاب المزيد، وانصح بالانتظار سنتين على الأقل بين الحملين.',
            source='كتاب1 D27',
            evidence_codes=('before_facility_discharge',),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='before_facility_discharge', value=True),
        ClinicalFact(code='breastfeeding', value=False),
        salience=80,
    )
    def d61_family_planning(self):
        self._follow4_emit(
            rule_id='D61',
            severity='low',
            category='family_planning',
            rule_type='Counseling',
            priority='Medium',
            confidence='High',
            title='تنظيم الأسرة لغير المرضعة',
            detail='اعرض فوراً الواقي أو وسائل البروجستيرون فقط أو الحقن أو الغرسة أو التعقيم أو اللولب النحاسي، وبعد 3 أسابيع الوسائل المركبة.',
            source='كتاب1 D27',
            evidence_codes=('breastfeeding', 'before_facility_discharge'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='before_facility_discharge', value=True),
        ClinicalFact(code='breastfeeding', value=True),
        salience=80,
    )
    def d62_family_planning(self):
        self._follow4_emit(
            rule_id='D62',
            severity='low',
            category='family_planning',
            rule_type='Counseling',
            priority='Medium',
            confidence='High',
            title='تنظيم الأسرة للمرضعة',
            detail='اعرض فوراً LAM والواقي والتعقيم أو اللولب النحاسي، وبعد 6 أسابيع البروجستيرون فقط أو الغرسة، وبعد 6 أشهر الوسائل المركبة.',
            source='كتاب1 D27',
            evidence_codes=('breastfeeding', 'before_facility_discharge'),
        )

    @Rule(
        ClinicalFact(code='in_delivery', value=True),
        ClinicalFact(code='delivery_phase_discharge', value=True),
        ClinicalFact(code='wants_lam_only', value=True),
        salience=110,
    )
    def d63_family_planning(self):
        self._follow4_emit(
            rule_id='D63',
            severity='medium',
            category='family_planning',
            rule_type='Assessment',
            priority='Medium',
            confidence='High',
            title='التحقق من شروط طريقة LAM',
            detail='تحقق من اجتماع الشروط الثلاثة: أقل من 6 أشهر بعد الولادة، رضاعة حصرية كاملة، وعدم عودة الدورة؛ وإلا أضف وسيلة أخرى فوراً.',
            source='كتاب1 D27',
            evidence_codes=('wants_lam_only', 'postpartum_under_6_months', 'exclusive_breastfeeding', 'amenorrhea'),
        )

