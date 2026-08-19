"""Fact-constrained SOAP presentation formatter.

This is intentionally *not* a second free-form LLM rewrite.  Language correction happens
once, before AraBERT, in the canonicalization stage.  The formatter only turns the
already-classified/canonical report items into clean doctor-facing section paragraphs.
Because every clause comes from a source item, it cannot invent a diagnosis, dose,
number, or negation that was not present in structured SOAP.
"""
from __future__ import annotations

import re
from typing import Mapping

from ..nlp.sections import SOAP_ORDER
from .schema import FormattedSoapSection, ReportSection

_WS = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s،؛;,.]+$")


class ClinicalSoapFormatter:
    applied = True
    name = "fact-constrained-v1"

    def format(self, soap: Mapping[str, ReportSection]) -> dict[str, FormattedSoapSection]:
        return {
            key: self.format_section(soap[key])
            for key in SOAP_ORDER
            if key in soap
        }

    def format_section(self, section: ReportSection) -> FormattedSoapSection:
        clauses: list[str] = []
        item_ids: list[str] = []
        seen: set[str] = set()
        for item in sorted(section.items, key=lambda row: row.order_index):
            text = _clean_clause(item.text)
            if not text:
                continue
            # Exact duplicate suppression is safe; semantic dedupe belongs in a model
            # only after it has its own fact-preservation validator.
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            clauses.append(text)
            item_ids.append(item.item_id)

        paragraph = "؛ ".join(clauses)
        if paragraph:
            paragraph += "."
        return FormattedSoapSection(
            soap_key=section.soap_key,
            title_ar=section.title_ar,
            text=paragraph,
            item_ids=item_ids,
        )


def _clean_clause(text: str) -> str:
    text = _WS.sub(" ", str(text or "")).strip()
    text = _TRAILING.sub("", text)
    if not text:
        return ""
    # Avoid lower-case/punctuation tricks; no lexical substitutions occur here.
    return text[0].upper() + text[1:] if text and text[0].isascii() else text
