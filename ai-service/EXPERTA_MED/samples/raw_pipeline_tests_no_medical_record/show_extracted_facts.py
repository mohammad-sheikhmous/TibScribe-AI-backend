from __future__ import annotations

import json
import sys
from pathlib import Path

from EXPERTA_MED.extraction import extract_report

HERE = Path(__file__).resolve().parent
files = [HERE / x for x in sys.argv[1:]] if len(sys.argv) > 1 else sorted(HERE.glob("raw_*.json"))

for path in files:
    report = json.loads(path.read_text(encoding="utf-8"))
    ext = extract_report(report, 0)

    print(f"\n=== {path.name} ===")
    print(
        f"context: pregnant={ext.pregnant}, labour={ext.labour}, "
        f"delivery={ext.delivery}, postpartum={ext.postpartum}, "
        f"ga={ext.ga_weeks}, labour_stage={ext.labour_stage}, "
        f"delivery_phase={ext.delivery_phase}"
    )
    for e in ext.entities:
        if e.negated:
            state = "NEGATED"
        else:
            state = "PRESENT"
        values = [f"value={e.value!r}"]
        if e.value2 is not None:
            values.append(f"value2={e.value2!r}")
        if e.unit:
            values.append(f"unit={e.unit}")
        print(f"- {e.kind}:{e.code} [{state}] {'; '.join(values)} | {e.text}")
