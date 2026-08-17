"""Validate the final source tree and, by default, the deployable model bundle.

Usage:
    python scripts/validate_final_build.py --source-only
    python scripts/validate_final_build.py              # deployment validation

`--source-only` is useful on a source archive before the separately trained model
checkpoint has been copied into model_output/. The normal/default mode fails if the
checkpoint is incomplete; production must never serve an untrained random head.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_LABELS = {
    "allergy", "cardiology", "diagnosis", "emergency", "follow_up", "gynecology",
    "history", "infection", "info", "lab", "medication", "neurology", "nutrition",
    "plan", "postpartum", "pregnancy_risk", "procedure", "symptom", "treatment", "vital",
}
RETIRED_FOLLOW5 = {f"R{i}" for i in range(28, 33)}
ACTIVE_FOLLOW5 = {f"R{i}" for i in range(1, 28)} | {f"R{i}" for i in range(33, 49)}


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def _rule_meta() -> tuple[set[str], set[str]]:
    path = ROOT / "EXPERTA_MED/md_rules/Pre_pregnancy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n.value for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "RULE_META" for t in n.targets)
    )
    meta = ast.literal_eval(node)
    emitted = {
        n.args[0].value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_follow5_emit"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)
    }
    return set(meta), emitted




def _emitted_rule_ids() -> set[str]:
    specs = [
        ("FOLLOW1", "follow1.py", "_follow1_emit"),
        ("FOLLOW2", "follow2.py", "_follow2_emit"),
        ("FOLLOW3", "labour.py", "_follow3_emit"),
        ("FOLLOW4", "birth.py", "_follow4_emit"),
        ("FOLLOW5", "Pre_pregnancy.py", "_follow5_emit"),
        ("FOLLOW6", "Infertility.py", "_follow6_emit"),
        ("FOLLOW7", "Postpartum.py", "_follow7_emit"),
    ]
    ids: set[str] = set()
    for prefix, filename, helper in specs:
        tree = ast.parse((ROOT / "EXPERTA_MED/md_rules" / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == helper
            ):
                continue
            value = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                value = node.args[0].value
            for keyword in node.keywords:
                if keyword.arg == "rule_id" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    value = keyword.value.value
            if value:
                ids.add(f"{prefix}-{value}")
    engine = (ROOT / "EXPERTA_MED/engine.py").read_text(encoding="utf-8")
    ids.update(re.findall(r'(?<!FOLLOW\d-)R\d{2}', engine))
    return ids

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []

    # Dataset and label contract.
    counts: Counter[str] = Counter()
    for line_no, line in enumerate((ROOT / "data.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            counts[str(row["label"])] += 1
        except Exception as exc:
            fail(f"data.jsonl line {line_no} invalid: {exc}", failures)
    if set(counts) == ACTIVE_LABELS and sum(counts.values()) == 16015:
        ok("dataset contract: 16,015 rows / 20 active labels")
    else:
        fail(
            f"dataset contract changed: rows={sum(counts.values())}, labels={sorted(counts)}",
            failures,
        )
    if "pregnancy_nutrition" not in counts:
        ok("retired pregnancy_nutrition label is absent")
    else:
        fail("retired pregnancy_nutrition label reappeared", failures)

    # Canonical KBS only.
    if not (ROOT / "EXPERTA_MED/rules").exists():
        ok("legacy YAML KBS is outside runtime")
    else:
        fail("legacy YAML KBS still exists under EXPERTA_MED/rules", failures)
    engine = (ROOT / "EXPERTA_MED/engine.py").read_text(encoding="utf-8")
    if re.search(r'RULES_VERSION\s*=\s*[\"\']0\.8\.0[\"\']', engine):
        ok("EXPERTA_MED rules version is 0.8.0")
    else:
        fail("unexpected EXPERTA_MED RULES_VERSION", failures)

    meta, emitted = _rule_meta()
    if meta == ACTIVE_FOLLOW5 and emitted == ACTIVE_FOLLOW5:
        ok("Follow5 stable IDs: R1-R27 + R33-R48")
    else:
        fail(f"Follow5 ID mismatch: meta={sorted(meta)} emit={sorted(emitted)}", failures)
    if not (meta & RETIRED_FOLLOW5) and not (emitted & RETIRED_FOLLOW5):
        ok("Follow5 R28-R32 are retired and not reused")
    else:
        fail("retired Follow5 R28-R32 are active/reused", failures)

    active_rule_ids = _emitted_rule_ids()
    if len(active_rule_ids) == 461:
        ok("KBS source contains 461 active rule IDs")
    else:
        fail(f"unexpected active KBS rule count: {len(active_rule_ids)}", failures)

    # Final integration expectation must not ask for retired IDs and every required
    # ID must actually exist in the current source tree.
    expected_text = (ROOT / "EXPERTA_MED/samples/final_integration_suite/expected_final_rules.json").read_text(encoding="utf-8")
    if not any(f"FOLLOW5-R{i}" in expected_text for i in range(28, 33)):
        ok("final KBS expectations exclude retired Follow5 IDs")
    else:
        fail("final KBS expectations still reference retired Follow5 IDs", failures)
    expected = json.loads(expected_text)
    required_ids = {
        rule_id
        for spec in expected.values()
        for field in ("must_include", "recommended_include", "allowed_cross_module_rules")
        for rule_id in spec.get(field, [])
    }
    unknown = sorted(required_ids - active_rule_ids)
    if not unknown:
        ok("all expected final-suite rule IDs exist in current KBS source")
    else:
        fail(f"final suite references unknown rules: {unknown}", failures)

    # A deployable model bundle is mandatory unless explicitly validating source only.
    model_dir = ROOT / "model_output"
    required = [model_dir / "best_model.pt", model_dir / "label_mapping.json", model_dir / "model_config.json"]
    if args.source_only:
        ok("source-only mode: trained model bundle check intentionally skipped")
    else:
        missing = [p.name for p in required if not p.is_file()]
        if missing:
            fail("model_output is not deployable; missing " + ", ".join(missing), failures)
        else:
            mapping = json.loads((model_dir / "label_mapping.json").read_text(encoding="utf-8"))
            labels = mapping.get("label2id") or {}
            if set(labels) == ACTIVE_LABELS and len(labels) == 20:
                ok("model label mapping matches the 20-label dataset")
            else:
                fail(f"model label mapping mismatch: {sorted(labels)}", failures)

    if failures:
        print(f"\nFINAL BUILD VALIDATION: FAIL ({len(failures)} problem(s))")
        return 1
    print("\nFINAL BUILD VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
