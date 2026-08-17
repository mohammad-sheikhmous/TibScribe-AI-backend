from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORBIDDEN_KEYS = {
    "medical_record",
    "clinical_facts",
    "structured_facts",
    "structured_data",
}

EXPECTED = json.loads(
    (HERE / "expected_final_rules.json").read_text(encoding="utf-8")
)

# Validate only the actual input reports listed in expected_final_rules.json.
# Generated files such as final_integration_run_summary.json must not be
# interpreted as AraBERT reports.
files = [HERE / filename for filename in EXPECTED]

failed = 0

def walk(value, path="$"):
    global failed
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                print(f"[FAIL] forbidden key {key!r} at {path}")
                failed += 1
            walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            walk(child, f"{path}[{i}]")

for file in files:
    if not file.exists():
        print(f"[FAIL] missing report: {file.name}")
        failed += 1
        continue

    data = json.loads(file.read_text(encoding="utf-8"))
    walk(data)

    item_count = 0
    for section in data.get("soap", {}).values():
        for item in section.get("items", []):
            item_count += 1
            if item.get("entity_links", "missing") is not None:
                print(f"[FAIL] {file.name}: entity_links must be null")
                failed += 1

    if item_count == 0:
        print(f"[FAIL] {file.name}: no SOAP items")
        failed += 1
    else:
        print(f"[PASS] {file.name}: {item_count} SOAP items")

print(f"\nValidated {len(files)} reports; failures={failed}")
raise SystemExit(1 if failed else 0)
