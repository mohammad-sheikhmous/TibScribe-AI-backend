"""Rephrase stage hook — the seam for a FUTURE LLM-based medical rewrite.

v1 ships only the no-op passthrough. A later stage plugs in by implementing the
RephraseStage protocol and populating ReportItem.text_rephrased — with NO change
to the Report schema, the builder signature, or the API layer.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import ReportSection


@runtime_checkable
class RephraseStage(Protocol):
    def rephrase_section(self, section: ReportSection) -> ReportSection:
        """Return the section, optionally with ReportItem.text_rephrased filled in."""
        ...

    @property
    def applied(self) -> bool:
        """Whether this stage actually rewrites text (drives pipeline_meta.rephrase_applied)."""
        ...


class NoOpRephraseStage:
    """Identity passthrough. Default for v1 — no LLM call happens here yet."""

    applied = False

    def rephrase_section(self, section: ReportSection) -> ReportSection:
        return section
