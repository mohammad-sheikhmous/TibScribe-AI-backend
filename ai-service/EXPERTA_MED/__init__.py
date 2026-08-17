"""EXPERTA_MED — Knowledge-Based System (Experta) over the medical-scribe reports.

Input : one Report JSON (the FastAPI service's output schema) or a chronological
        series of reports tracking the same patient.
Output: explainable clinical suggestions (missing questions/tests, escalations)
        with rule provenance, inference chains, and an audit trail.
"""
ENGINE_VERSION = "0.8.0"
