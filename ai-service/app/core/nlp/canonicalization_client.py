"""LLM client for pre-AraBERT medical canonicalization.

This module contains the provider-specific communication layer.

It does NOT decide whether a correction is medically safe.
Safety validation remains the responsibility of canonicalization.py.

Flow:

    Segment batch
        ↓
    OpenAICanonicalizationClient
        ↓
    OpenAI Responses API
        ↓
    Structured JSON response
        ↓
    {segment_id: corrected_text}
        ↓
    LLMMedicalCanonicalizationStage
        ↓
    Safety guards
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from .canonicalization import CanonicalizationClient


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured response models
# ---------------------------------------------------------------------------


class CanonicalizedItem(BaseModel):
    """One corrected segment returned by the LLM."""

    segment_id: str

    corrected_text: str


class CanonicalizationResponse(BaseModel):
    """Expected structured response from the LLM."""

    items: list[CanonicalizedItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


MEDICAL_CANONICALIZATION_INSTRUCTIONS = """
أنت طبقة تصحيح لغوي طبي تعمل ضمن نظام تدوين طبي متخصص في طب النساء
والتوليد والحمل.

مهمتك الوحيدة هي تحويل النص العربي الناتج عن التعرف الصوتي إلى عربية
طبية واضحة وسليمة لغويًا، مع الحفاظ التام على المعنى الطبي الأصلي.

قد يحتوي النص على:
- لهجة عربية عامية.
- أخطاء إملائية.
- أخطاء واضحة ناتجة عن التعرف الصوتي ASR.
- مصطلحات طبية مكتوبة بطريقة غير دقيقة.
- جمل غير مرتبة لغويًا.
- كلمات عربية وأجنبية أو اختصارات طبية.

المسموح لك:
1. تصحيح الأخطاء الإملائية.
2. تصحيح الأخطاء النحوية الواضحة.
3. تحويل التعبير العامي إلى عربية فصحى طبية طبيعية.
4. تصحيح مصطلح طبي مشوه إذا كان التصحيح واضحًا جدًا من السياق.
5. ترتيب الجملة لغويًا دون تغيير معناها.
6. توحيد الصياغة الطبية.
7. إزالة الحشو الكلامي غير الدلالي مثل:
   "يعني"، "طيب"، "مثلاً"
   فقط عندما لا يحمل أي معنى طبي.
8. الاستفادة من سياق المقاطع المجاورة لفهم الكلمة، ولكن دون نقل
   معلومات من مقطع إلى مقطع آخر.

ممنوع منعًا باتًا:
1. إضافة معلومة طبية لم تذكر في النص.
2. إضافة تشخيص أو استنتاج طبي.
3. حذف معلومة طبية موجودة.
4. تغيير أي رقم.
5. تغيير ضغط الدم.
6. تغيير درجة الحرارة.
7. تغيير النبض.
8. تغيير عمر الحمل.
9. تغيير تاريخ أو عمر المريضة.
10. تغيير قيمة تحليل مخبري.
11. تغيير جرعة دواء.
12. تغيير وحدة قياس.
13. تغيير اسم دواء إلى دواء آخر.
14. تحويل الشك أو الاحتمال إلى تشخيص مؤكد.
15. تحويل التشخيص المؤكد إلى احتمال.
16. تغيير النفي إلى الإثبات أو العكس.
17. تحويل سؤال الطبيب إلى حقيقة طبية.
18. نقل معلومة من segment إلى segment آخر.
19. دمج المقاطع أو تقسيمها.
20. تغيير segment_id.

إذا كانت كلمة مشوهة ويمكن تصحيحها طبيًا بثقة عالية جدًا، صححها.

مثال:
"قلم بالبطن"
يمكن أن تصبح:
"ألم في البطن"
إذا كان السياق واضحًا.

مثال:
"الجنين بوضعية رئيسية"
يمكن أن تصبح:
"الجنين بوضعية رأسية"
فقط عندما يكون السياق التوليدي واضحًا.

أما إذا كان التصحيح غير مؤكد، احتفظ بالكلمة أو العبارة الأصلية ولا تخمن.

أمثلة صحيحة:

Input:
"في عندها وجع راس خفيف"

Output:
"تعاني من صداع خفيف."

Input:
"رجليها شوي متنفخين"

Output:
"تعاني من تورم بسيط في القدمين."

Input:
"ما عندها نزيف"

Output:
"لا تعاني من نزيف."

Input:
"ضغطها طلع 128 على 82"

Output:
"بلغ ضغط الدم 128 على 82."

مثال ممنوع:

Input:
"ضغطها 128 على 82"

Output:
"ضغطها 138 على 92"

هذا ممنوع لأن الأرقام تغيرت.

مثال ممنوع:

Input:
"لا يوجد نزيف"

Output:
"تعاني من نزيف"

هذا ممنوع لأن النفي تغير.

أعد عنصرًا واحدًا مقابل كل عنصر دخل إليك، وحافظ على segment_id نفسه.
لا تكتب شرحًا أو ملاحظات أو توصيات.
أعد فقط البيانات المطلوبة في البنية المحددة.
""".strip()


# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------


_CANONICALIZATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                    },
                    "corrected_text": {
                        "type": "string",
                    },
                },
                "required": [
                    "segment_id",
                    "corrected_text",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAICanonicalizationClient(CanonicalizationClient):
    """OpenAI implementation of CanonicalizationClient.

    The class only communicates with the LLM and parses its structured output.

    It deliberately does NOT perform clinical safety validation.

    Safety validation happens later in:

        LLMMedicalCanonicalizationStage

    inside canonicalization.py.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_sec: float = 20.0,
        base_url: str | None = None,
    ):
        if not api_key.strip():
            raise ValueError(
                "canonicalization API key must not be empty"
            )

        if not model.strip():
            raise ValueError(
                "canonicalization model must not be empty"
            )

        self.model = model.strip()

        client_kwargs: dict[str, Any] = {
            "api_key": api_key.strip(),
            "timeout": timeout_sec,
        }

        if base_url and base_url.strip():
            client_kwargs["base_url"] = base_url.strip()

        self.client = OpenAI(**client_kwargs)

    def canonicalize_batch(
        self,
        items: list[dict[str, str]],
    ) -> dict[str, str]:
        """Canonicalize a batch of transcript segments.

        Args:
            items:
                Example:

                [
                    {
                        "segment_id": "0",
                        "speaker": "patient",
                        "text": "رجليها شوي متنفخين"
                    }
                ]

        Returns:
            Mapping:

                {
                    "0": "تعاني من تورم بسيط في القدمين."
                }

        Raises:
            Any API/parsing exception is intentionally allowed to propagate.

            canonicalization.py catches it and safely falls back to the
            original ASR text.
        """

        if not items:
            return {}

        payload = self._build_payload(items)

        response = self.client.responses.create(
            model=self.model,

            instructions=MEDICAL_CANONICALIZATION_INSTRUCTIONS,

            input=json.dumps(
                payload,
                ensure_ascii=False,
            ),

            text={
                "format": {
                    "type": "json_schema",
                    "name": "medical_canonicalization",
                    "strict": True,
                    "schema": _CANONICALIZATION_JSON_SCHEMA,
                }
            },
        )

        output_text = response.output_text

        if not output_text:
            raise RuntimeError(
                "LLM returned an empty canonicalization response"
            )

        parsed = CanonicalizationResponse.model_validate_json(
            output_text
        )

        result = self._validate_response_ids(
            input_items=items,
            response=parsed,
        )

        return result

    # -----------------------------------------------------------------------
    # Request preparation
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_payload(
        items: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Build a minimal, explicit payload for the model."""

        prepared_items: list[dict[str, str]] = []

        for item in items:
            segment_id = str(
                item.get("segment_id", "")
            ).strip()

            text = str(
                item.get("text", "")
            ).strip()

            speaker = str(
                item.get("speaker", "unknown")
            ).strip() or "unknown"

            if not segment_id:
                raise ValueError(
                    "canonicalization item is missing segment_id"
                )

            if not text:
                # Empty text has nothing useful to canonicalize.
                # Keeping it in the request is unnecessary.
                continue

            prepared_items.append(
                {
                    "segment_id": segment_id,
                    "speaker": speaker,
                    "text": text,
                }
            )

        return {
            "task": (
                "Canonicalize each Arabic medical transcript segment "
                "without changing its clinical meaning."
            ),
            "items": prepared_items,
        }

    # -----------------------------------------------------------------------
    # Response verification
    # -----------------------------------------------------------------------

    @staticmethod
    def _validate_response_ids(
        *,
        input_items: list[dict[str, str]],
        response: CanonicalizationResponse,
    ) -> dict[str, str]:
        """Verify segment identity before returning LLM output.

        Structured Outputs guarantee the JSON shape, but the application
        still owns semantic checks such as:

        - duplicate IDs;
        - unknown IDs;
        - missing IDs.

        Missing items are allowed here because canonicalization.py will
        automatically fall back to the original ASR text for them.
        """

        expected_ids = {
            str(item["segment_id"])
            for item in input_items
            if item.get("segment_id") is not None
        }

        result: dict[str, str] = {}

        for item in response.items:
            segment_id = str(item.segment_id).strip()

            corrected_text = item.corrected_text.strip()

            if segment_id not in expected_ids:
                logger.warning(
                    "Canonicalization LLM returned unknown "
                    "segment_id=%s; ignoring it",
                    segment_id,
                )

                continue

            if segment_id in result:
                logger.warning(
                    "Canonicalization LLM returned duplicate "
                    "segment_id=%s; keeping first result",
                    segment_id,
                )

                continue

            if not corrected_text:
                logger.warning(
                    "Canonicalization LLM returned empty text "
                    "for segment_id=%s",
                    segment_id,
                )

                continue

            result[segment_id] = corrected_text

        missing_ids = expected_ids - set(result)

        if missing_ids:
            logger.warning(
                "Canonicalization LLM omitted %d segment(s): %s",
                len(missing_ids),
                sorted(missing_ids),
            )

        return result