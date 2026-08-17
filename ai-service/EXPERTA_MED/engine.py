"""MedicalKBS — the Experta KnowledgeEngine.

Facts (from extraction.py) are declared into working memory; rules fire to produce
explainable Suggestions. Each rule is versioned and, when it fires, records:
  - the evidence facts (with item_ids) that matched, and
  - an Arabic inference chain (fact -> rule -> suggestion),
so every suggestion is auditable back to a report sentence and its audio timestamp.

Missing-test detection is driven by the knowledge graph: for a suspected/known
condition, `graph.required_tests(cond)` minus the tests already seen in the reports
= the gap the doctor is prompted about.

Clinical rules are curated from standard OB/GYN practice and NEED CLINICIAN
SIGN-OFF before real use.
"""
from __future__ import annotations

from . import _compat  # noqa: F401
from experta import AS, MATCH, NOT, Rule, KnowledgeEngine

from .facts import (
    Allergy, DiagnosisNoted, EmergencyFlag, FollowUpPlanned, LabResult,
    LowConfidenceCritical, Medication, PatientContext, ReportMeta, Symptom,
    TestDone, Trend, Vital,
)
from .knowledge_graph import KnowledgeGraph
from .ontology import ar, drug_family
from .suggestions import Suggestion, merge_suggestions
from .md_rules.follow1 import Follow1Rules
from .md_rules.follow2 import Follow2Rules
from .md_rules.labour import Follow3Rules
from .md_rules.birth import Follow4Rules
from .md_rules.Pre_pregnancy import Follow5Rules
from .md_rules.Infertility import Follow6Rules
from .md_rules.Postpartum import Follow7Rules

RULES_VERSION = "0.8.0"


class MedicalKBS(Follow7Rules, Follow6Rules, Follow5Rules, Follow4Rules, Follow3Rules, Follow2Rules, Follow1Rules, KnowledgeEngine):
    def __init__(self, graph: KnowledgeGraph | None = None):
        super().__init__()
        self.graph = graph or KnowledgeGraph()
        self.suggestions: list[Suggestion] = []
        self.fired: list[dict] = []           # audit trail
        self._tests_done: set[str] = set()
        self._emitted: set[tuple] = set()      # dedupe key per (rule_id, condition)
        self._emitted_by_key: dict[tuple, Suggestion] = {}

    # ---- helpers -----------------------------------------------------------
    def _audit(self, rule_id: str, version: str, evidence: list[dict]):
        self.fired.append({"rule_id": rule_id, "version": version, "evidence": evidence})

    def _missing_tests(self, condition: str) -> list[str]:
        return [t for t in self.graph.required_tests(condition) if t not in self._tests_done]

    def _emit(self, sug: Suggestion, dedupe: tuple):
        # Deduplication is lossless: if the same clinical recommendation fires from
        # multiple independent facts, keep ONE suggestion but merge all provenance.
        if dedupe in self._emitted:
            current = self._emitted_by_key[dedupe]
            merge_suggestions(current, sug)
            self._audit(sug.rule_id, sug.rule_version, sug.evidence)
            return

        self._emitted.add(dedupe)
        # Normalize duplicate list entries even on the first firing.
        merge_suggestions(sug, Suggestion(
            rule_id=sug.rule_id, rule_version=sug.rule_version, severity=sug.severity,
            title_ar=sug.title_ar, detail_ar=sug.detail_ar
        ))
        self._emitted_by_key[dedupe] = sug
        self.suggestions.append(sug)
        self._audit(sug.rule_id, sug.rule_version, sug.evidence)

    # ---- bookkeeping: collect all performed tests first (high salience) -----
    @Rule(TestDone(code=MATCH.c), salience=100)
    def _collect_tests(self, c):
        self._tests_done.add(c)

    # =======================================================================
    # PREECLAMPSIA cluster
    # =======================================================================
    @Rule(
        PatientContext(pregnant=True),
        AS.s1 << Symptom(code="headache", negated=False),
        AS.s2 << Symptom(code="blurred_vision", negated=False),
        salience=20,
    )
    def r01_preeclampsia_symptom_cluster(self, s1, s2):
        """R01 v0.1 — حامل + صداع + تشوش رؤية → اشتباه تسمم حمل، افحص الضغط/الزلال."""
        missing = self._missing_tests("preeclampsia")
        ev = [_ev(s1), _ev(s2)]
        self._emit(Suggestion(
            rule_id="R01", rule_version=RULES_VERSION, severity="high",
            condition="preeclampsia",
            title_ar="اشتباه تسمم حمل — فحوصات ناقصة",
            detail_ar="اجتماع الصداع وتشوش الرؤية لدى حامل يرفع احتمال تسمم الحمل؛ "
                      "يُنصح باستكمال الفحوصات أدناه إن لم تُجرَ.",
            missing_tests=missing, evidence=ev,
            chain=[
                "حامل (PatientContext.pregnant=صحيح)",
                f"عرض: {ar('headache')} + {ar('blurred_vision')} (الرسم البياني: indicates → {ar('preeclampsia')})",
                "R01 → اقتراح استكمال فحوصات تسمم الحمل الناقصة",
            ],
        ), dedupe=("R01", "preeclampsia"))

    @Rule(
        PatientContext(pregnant=True),
        AS.v << Vital(kind="bp", sys=MATCH.sys),
        NOT(LabResult(code="urine_protein")),
        NOT(TestDone(code="urine_protein_test")),
        salience=20,
    )
    def r02_high_bp_no_protein(self, v, sys):
        """R02 v0.1 — حامل بضغط انقباضي ≥140 دون تحليل زلال → اطلب تحليل الزلال."""
        if float(sys) < 140:
            return
        ev = [_ev(v)]
        self._emit(Suggestion(
            rule_id="R02", rule_version=RULES_VERSION, severity="high",
            condition="preeclampsia", missing_tests=["urine_protein_test"],
            title_ar="ضغط مرتفع بلا تحليل زلال البول",
            detail_ar=f"الضغط الانقباضي {int(float(sys))} مرتفع لدى حامل ولم يُذكر تحليل زلال البول — "
                      "أساسي لتشخيص/استبعاد تسمم الحمل.",
            evidence=ev,
            chain=[f"قياس ضغط انقباضي = {int(float(sys))} (≥140)",
                   "لا يوجد LabResult(urine_protein) ولا TestDone(urine_protein_test)",
                   "R02 → اطلب تحليل زلال البول"],
        ), dedupe=("R02", "bp"))

    @Rule(
        AS.d << DiagnosisNoted(code="preeclampsia"),
        salience=15,
    )
    def r03_preeclampsia_workup_gap(self, d):
        """R03 v0.1 — تسمم حمل مشخّص → أكمل الفحوصات القياسية الناقصة."""
        missing = self._missing_tests("preeclampsia")
        if not missing:
            return
        self._emit(Suggestion(
            rule_id="R03", rule_version=RULES_VERSION, severity="high",
            condition="preeclampsia", missing_tests=missing,
            title_ar="تسمم حمل مشخّص — استكمال الفحوصات القياسية",
            detail_ar="عند تشخيص تسمم الحمل، الفحوصات التالية جزء من التقييم القياسي ولم تُذكر.",
            evidence=[_ev(d)],
            chain=[f"تشخيص مذكور: {ar('preeclampsia')}",
                   f"من الرسم البياني requires_test، الناقص: {'، '.join(ar(t) for t in missing)}",
                   "R03 → استكمال الفحوصات"],
        ), dedupe=("R03", "preeclampsia"))

    # =======================================================================
    # GDM cluster
    # =======================================================================
    @Rule(AS.d << DiagnosisNoted(code="gdm"),
          NOT(LabResult(code="hba1c")), NOT(TestDone(code="hba1c_test")), salience=15)
    def r04_gdm_no_hba1c(self, d):
        """R04 v0.1 — سكري حمل دون سكر تراكمي → اطلب HbA1c."""
        self._emit(Suggestion(
            rule_id="R04", rule_version=RULES_VERSION, severity="medium",
            condition="gdm", missing_tests=self._missing_tests("gdm"),
            title_ar="سكري حمل — سكر تراكمي غير مذكور",
            detail_ar="لضبط سكري الحمل يُنصح بمعايرة HbA1c وسكر الصائم إن لم تُجرَ.",
            evidence=[_ev(d)],
            chain=[f"تشخيص: {ar('gdm')}", "لا HbA1c ولا سكر صائم مذكور", "R04 → اطلب المعايرة"],
        ), dedupe=("R04", "gdm"))

    # =======================================================================
    # Fetal wellbeing
    # =======================================================================
    @Rule(PatientContext(pregnant=True),
          AS.s << Symptom(code="reduced_fetal_movement", negated=False),
          NOT(TestDone(code="ctg")), salience=20)
    def r05_reduced_movement_no_ctg(self, s):
        """R05 v0.1 — قلة حركة الجنين دون CTG → اطلب مراقبة نبض الجنين."""
        self._emit(Suggestion(
            rule_id="R05", rule_version=RULES_VERSION, severity="high",
            condition="preterm_labor", missing_tests=["ctg", "obstetric_ultrasound"],
            title_ar="قلة حركة الجنين — تقييم غير مكتمل",
            detail_ar="قلة حركة الجنين تستوجب مراقبة نبض الجنين (CTG) وتصويراً توليدياً لتقييم السلامة.",
            evidence=[_ev(s)],
            chain=[f"عرض: {ar('reduced_fetal_movement')}", "لا CTG مُجرى", "R05 → CTG + سونار"],
        ), dedupe=("R05", "fetal"))

    # =======================================================================
    # Allergy / medication safety (join across facts)
    # =======================================================================
    @Rule(AS.a << Allergy(family=MATCH.fam),
          AS.m << Medication(family=MATCH.fam), salience=30)
    def r06_allergy_medication_conflict(self, a, m, fam):
        """R06 v0.1 — دواء مصروف من عائلة تتحسس منها المريضة → تعارض دوائي."""
        self._emit(Suggestion(
            rule_id="R06", rule_version=RULES_VERSION, severity="critical",
            title_ar="⚠️ تعارض حساسية دوائية",
            detail_ar=f"صُرف دواء من عائلة ({fam}) تتحسس منها المريضة — راجعي البديل فوراً.",
            evidence=[_ev(a), _ev(m)],
            chain=[f"حساسية: {a['drug'] if isinstance(a, dict) else ''}",
                   "دواء مصروف من نفس العائلة", "R06 → تنبيه تعارض"],
        ), dedupe=("R06", fam))

    # =======================================================================
    # Infection workup
    # =======================================================================
    @Rule(AS.d << DiagnosisNoted(code="infection"),
          NOT(TestDone(code="cbc")), salience=12)
    def r07_infection_no_cbc(self, d):
        """R07 v0.1 — عدوى دون تعداد دم → اطلب CBC (± زرع)."""
        self._emit(Suggestion(
            rule_id="R07", rule_version=RULES_VERSION, severity="medium",
            condition="infection", missing_tests=self._missing_tests("infection"),
            title_ar="عدوى — تعداد الدم غير مذكور",
            detail_ar="عند وجود عدوى يُنصح بتعداد دم كامل وزرع لتوجيه المضاد الحيوي.",
            evidence=[_ev(d)],
            chain=[f"تشخيص: {ar('infection')}", "لا CBC", "R07 → CBC + زرع"],
        ), dedupe=("R07", "infection"))

    @Rule(PatientContext(postpartum=True),
          AS.s << Symptom(code="fever", negated=False), salience=18)
    def r08_postpartum_fever(self, s):
        """R08 v0.1 — حرارة بعد الولادة → استبعاد إنتان النفاس."""
        self._emit(Suggestion(
            rule_id="R08", rule_version=RULES_VERSION, severity="high",
            condition="puerperal_infection",
            missing_tests=self._missing_tests("puerperal_infection"),
            title_ar="حرارة بعد الولادة — اشتباه إنتان نفاس",
            detail_ar="الحرارة في فترة النفاس تستوجب استبعاد إنتان النفاس (فحص الجرح + CBC + زرع دم).",
            evidence=[_ev(s)],
            chain=[f"سياق: ما بعد الولادة", f"عرض: {ar('fever')}", "R08 → تقييم إنتان النفاس"],
        ), dedupe=("R08", "puerperal"))

    # =======================================================================
    # Emergency escalation
    # =======================================================================
    @Rule(AS.e << EmergencyFlag(), salience=50)
    def r09_emergency_escalation(self, e):
        """R09 v0.1 — عنصر طارئ → تصعيد فوري."""
        self._emit(Suggestion(
            rule_id="R09", rule_version=RULES_VERSION, severity="critical",
            title_ar="🚨 تصعيد فوري — عنصر طارئ",
            detail_ar="رُصد عنصر مُصنَّف طارئاً — يتطلب تقييماً/تدخلاً عاجلاً وتوثيقاً للإجراء.",
            evidence=[_ev(e)],
            chain=["عنصر EmergencyFlag موجود", "R09 → تصعيد فوري"],
        ), dedupe=("R09", "emergency_escalation"))

    # =======================================================================
    # Data-quality / safety net
    # =======================================================================
    @Rule(AS.lc << LowConfidenceCritical(label=MATCH.lbl), salience=8)
    def r10_low_confidence_review(self, lc, lbl):
        """R10 v0.1 — تصنيف حرج منخفض الثقة → يحتاج تأكيد بشري."""
        self._emit(Suggestion(
            rule_id="R10", rule_version=RULES_VERSION, severity="low",
            title_ar="عنصر حرج منخفض الثقة — يحتاج تأكيد",
            detail_ar=f"عنصر مُصنَّف «{lbl}» بثقة منخفضة؛ أكّدي التصنيف قبل الاعتماد عليه في القرار.",
            evidence=[_ev(lc)],
            chain=[f"تصنيف حرج ({lbl}) + ثقة منخفضة", "R10 → مراجعة بشرية"],
        ), dedupe=("R10", lc.get("item_id", "")))

    # =======================================================================
    # General internal medicine (drug safety / chronic conditions)
    # =======================================================================
    @Rule(AS.m << Medication(family="nsaid"),
          AS.d << DiagnosisNoted(code="hypertension"), salience=30)
    def r13_nsaid_in_hypertension(self, m, d):
        """R13 v0.1 — مضاد التهاب (NSAID) لمريض ارتفاع ضغط → تحذير دوائي."""
        self._emit(Suggestion(
            rule_id="R13", rule_version=RULES_VERSION, severity="high",
            condition="hypertension",
            title_ar="⚠️ مضاد التهاب (NSAID) لمريض ارتفاع ضغط",
            detail_ar="صُرف مضاد التهاب غير ستيرويدي (مثل بروفين/إيبوبروفين) لمريض ارتفاع ضغط الدم — "
                      "قد يرفع الضغط ويُضعف فعالية خافضات الضغط. راجعي البديل (باراسيتامول) والجرعة/المدة.",
            evidence=[_ev(m), _ev(d)],
            chain=["دواء من عائلة NSAID مصروف",
                   f"المريض لديه {ar('hypertension')}",
                   "R13 → تحذير: NSAID يرفع الضغط + مراجعة البديل"],
        ), dedupe=("R13", "htn"))

    @Rule(AS.d << DiagnosisNoted(code="hypertension"),
          AS.v << Vital(kind="bp", sys=MATCH.sys), salience=13)
    def r14_uncontrolled_bp(self, d, v, sys):
        """R14 v0.1 — ارتفاع ضغط معروف + قياس انقباضي ≥140 → الضغط غير مضبوط."""
        if float(sys) < 140:
            return
        self._emit(Suggestion(
            rule_id="R14", rule_version=RULES_VERSION, severity="medium",
            condition="hypertension",
            title_ar="ضغط الدم غير مضبوط",
            detail_ar=f"المريض معروف بارتفاع ضغط الدم والقياس الحالي ({int(float(sys))}) ما زال مرتفعاً — "
                      "راجعي ضبط العلاج والالتزام به.",
            evidence=[_ev(v), _ev(d)],
            chain=[f"تشخيص/تاريخ: {ar('hypertension')}",
                   f"قياس انقباضي حالي = {int(float(sys))} (≥140)",
                   "R14 → مراجعة ضبط الضغط"],
        ), dedupe=("R14", "htn"))

    # =======================================================================
    # Temporal rules (fed by temporal.py as Trend facts)
    # =======================================================================
    @Rule(AS.t << Trend(kind="persistent", code=MATCH.c, count=MATCH.n), salience=14)
    def r11_persistent_symptom(self, t, c, n):
        """R11 v0.1 — عرض مستمر عبر عدة تقارير → يستدعي تقييماً أعمق."""
        self._emit(Suggestion(
            rule_id="R11", rule_version=RULES_VERSION, severity="medium",
            condition=None,
            title_ar=f"عرض مستمر عبر {n} زيارات: {ar(c)}",
            detail_ar=f"العرض «{ar(c)}» تكرر في {n} تقارير متتالية دون تحسّن موثّق — يستدعي تقييماً أعمق.",
            evidence=[{"item_id": i, "text": tx, "report_index": None}
                      for i, tx in zip(t.get("evidence_ids", ()), t.get("evidence_texts", ()))],
            chain=[f"العرض {ar(c)} ظهر في {n} تقارير", "R11 → تقييم أعمق"],
        ), dedupe=("R11", c))

    @Rule(AS.t << Trend(kind="worsening", code="bp", delta=MATCH.d), salience=22)
    def r12_worsening_bp(self, t, d):
        """R12 v0.1 — ضغط الدم يتصاعد عبر الزيارات → خطر تفاقم."""
        self._emit(Suggestion(
            rule_id="R12", rule_version=RULES_VERSION, severity="high",
            condition="preeclampsia",
            title_ar="تصاعد ضغط الدم عبر الزيارات",
            detail_ar=f"الضغط الانقباضي ارتفع بمقدار {int(float(d))} تقريباً بين الزيارات — راقبي تفاقم تسمم الحمل.",
            evidence=[{"item_id": i, "text": tx, "report_index": None}
                      for i, tx in zip(t.get("evidence_ids", ()), t.get("evidence_texts", ()))],
            chain=[f"ضغط انقباضي متصاعد (+{int(float(d))})", "R12 → مراقبة تفاقم"],
        ), dedupe=("R12", "bp_trend"))


def _ev(fact) -> dict:
    """Extract provenance fields from a matched fact (Experta Fact behaves dict-like)."""
    g = fact.get if hasattr(fact, "get") else (lambda k, d=None: d)
    return {
        "item_id": g("item_id", ""),
        "text": g("text", ""),
        "report_index": g("report_index", None),
        "drug": g("drug", None),
    }
