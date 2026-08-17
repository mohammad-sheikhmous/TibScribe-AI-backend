from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# When this file is executed by path, Python places this samples
# directory on sys.path rather than the project parent.
PROJECT_PARENT = Path(__file__).resolve().parents[3]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from EXPERTA_MED.extraction import extract_report

HERE = Path(__file__).resolve().parent

for path in sorted(HERE.glob("final_*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    ext = extract_report(report, 0)

    codes = [
        e.code
        for e in ext.entities
        if not getattr(e, "negated", False)
    ]
    module_triggers = Counter()
    for code in codes:
        if code.startswith("follow5_"):
            module_triggers["FOLLOW5"] += 1
        elif code.startswith("follow6_"):
            module_triggers["FOLLOW6"] += 1
        elif code.startswith("follow7_"):
            module_triggers["FOLLOW7"] += 1

    print(f"\n=== {path.name} ===")
    print(
        "context:",
        f"pregnant={ext.pregnant}",
        f"labour={ext.labour}",
        f"delivery={ext.delivery}",
        f"postpartum={ext.postpartum}",
        f"labour_stage={ext.labour_stage}",
        f"delivery_phase={ext.delivery_phase}",
        f"ga={ext.ga_weeks}",
    )
    print("entities:", len(ext.entities))
    print("source trigger counts:", dict(module_triggers))
    print("codes:", ", ".join(sorted(set(codes))))
