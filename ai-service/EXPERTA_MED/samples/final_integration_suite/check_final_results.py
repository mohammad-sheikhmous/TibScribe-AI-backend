from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT_PARENT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = DEFAULT_PROJECT_PARENT / "EXPERTA_MED" / "output"
EXPECTED = json.loads(
    (HERE / "expected_final_rules.json").read_text(encoding="utf-8")
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--warnings-as-errors",
    action="store_true",
    help="Fail when recommended rules are missing.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT,
    help="Directory containing *.suggestions.json files.",
)
args = parser.parse_args()
OUTPUT = args.output_dir.resolve()

failed = 0
warning_count = 0
summary = {}

for filename, spec in EXPECTED.items():
    stem = Path(filename).stem
    output_file = OUTPUT / f"{stem}.suggestions.json"

    if not output_file.exists():
        print(f"[MISSING OUTPUT] {filename}")
        failed += 1
        continue

    data = json.loads(output_file.read_text(encoding="utf-8"))
    suggestions = data.get("suggestions", [])
    fired = {
        item.get("rule_id")
        for item in suggestions
        if isinstance(item.get("rule_id"), str)
    }

    missing = [rule for rule in spec["must_include"] if rule not in fired]
    forbidden = [rule for rule in spec["must_not_include"] if rule in fired]

    allowed_cross_module = set(spec.get("allowed_cross_module_rules", []))
    leaked = sorted(
        rule for rule in fired
        if rule not in allowed_cross_module
        and any(
            rule.startswith(prefix)
            for prefix in spec["forbidden_prefixes"]
        )
    )

    absent_prefixes = [
        prefix
        for prefix in spec["required_prefixes"]
        if not any(rule.startswith(prefix) for rule in fired)
    ]

    severity_counts = Counter(
        str(item.get("severity", "")).lower()
        for item in suggestions
    )
    critical_missing = (
        severity_counts.get("critical", 0)
        < spec.get("minimum_critical", 0)
    )

    recommended_missing = [
        rule for rule in spec.get("recommended_include", [])
        if rule not in fired
    ]

    hard_fail = bool(
        missing
        or forbidden
        or leaked
        or absent_prefixes
        or critical_missing
    )

    if hard_fail:
        print(f"[FAIL] {filename}")
        if missing:
            print("  missing required:", ", ".join(missing))
        if forbidden:
            print("  unexpected rules:", ", ".join(forbidden))
        if leaked:
            print("  stage/module leakage:", ", ".join(leaked))
        if absent_prefixes:
            print("  missing required modules:", ", ".join(absent_prefixes))
        if critical_missing:
            print(
                "  critical suggestions:",
                severity_counts.get("critical", 0),
                "<",
                spec.get("minimum_critical", 0),
            )
        failed += 1
    else:
        print(
            f"[PASS] {filename} "
            f"({len(suggestions)} suggestions; "
            f"{severity_counts.get('critical', 0)} critical)"
        )

    if recommended_missing:
        print(
            "  [WARN] recommended but absent:",
            ", ".join(recommended_missing),
        )
        warning_count += len(recommended_missing)

    modules = Counter()
    for rule in fired:
        if rule.startswith("FOLLOW"):
            modules[rule.split("-", 1)[0]] += 1
        elif rule.startswith("R"):
            modules["LEGACY"] += 1

    summary[filename] = {
        "scenario": spec["scenario"],
        "suggestions_count": len(suggestions),
        "severity_counts": dict(severity_counts),
        "module_counts": dict(modules),
        "missing_required": missing,
        "unexpected_rules": forbidden,
        "module_leakage": leaked,
        "allowed_cross_module_rules": sorted(
            rule for rule in fired if rule in allowed_cross_module
        ),
        "missing_required_modules": absent_prefixes,
        "recommended_missing": recommended_missing,
        "passed": not hard_fail,
    }

(HERE / "final_integration_run_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print()
print(
    f"Scenarios: {len(EXPECTED)} | "
    f"Failed: {failed} | Warnings: {warning_count}"
)
print("Output directory:", OUTPUT)
print(
    "Summary written to:",
    HERE / "final_integration_run_summary.json",
)

if args.warnings_as_errors and warning_count:
    failed += 1

raise SystemExit(1 if failed else 0)
