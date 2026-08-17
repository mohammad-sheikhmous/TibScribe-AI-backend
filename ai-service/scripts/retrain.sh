#!/usr/bin/env bash
# Periodic self-improvement cycle (IMPLEMENTATION.md P7-09 / P7-11).
#
# Runs the whole loop: harvest what the doctors corrected, re-adapt the encoder on the
# transcripts that accumulated, retrain, and — only if it is actually better — activate.
#
# The gate is the point. A retrain that scores worse than the model in production must
# not reach production, and "worse" is decided on the gold set of real transcripts, not
# on the synthetic split that every model does well on.
#
#   bash scripts/retrain.sh                 # full cycle, activates only if better
#   bash scripts/retrain.sh --no-pseudo     # human corrections only
#   bash scripts/retrain.sh --dry-run       # show what would run
set -euo pipefail

PY="${PYTHON:-.venv/Scripts/python}"
GOLD_DIR="${GOLD_DIR:-eval/gold}"
CORRECTIONS="data/gold/corrections.jsonl"
PSEUDO="data/gold/pseudo.jsonl"
CANDIDATE_DIR="model_output_candidate"
ACTIVE_DIR="model_output"
REPORT="reports/retrain_$(date -u +%Y%m%dT%H%M%SZ).json"

USE_PSEUDO=1
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-pseudo) USE_PSEUDO=0 ;;
    --dry-run)   DRY_RUN=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

run() {
  echo "+ $*"
  [ "$DRY_RUN" -eq 1 ] || "$@"
}

echo "=== 1/6  harvest doctor corrections ==========================================="
run "$PY" scripts/export_training_data.py --out "$CORRECTIONS" --include-confirmed

echo "=== 2/6  guarded pseudo-labels ================================================"
if [ "$USE_PSEUDO" -eq 1 ]; then
  run "$PY" scripts/pseudo_label.py --out "$PSEUDO"
else
  echo "  (skipped: --no-pseudo)"
fi

echo "=== 3/6  re-adapt the encoder on accumulated transcripts (DAPT) ==============="
# Unsupervised: every stored transcript is corpus, no labelling needed (PLAN_V2 §5).
run "$PY" train_mlm.py --extra-texts data/results --output-dir model_output/bert_adapted

echo "=== 4/6  train a CANDIDATE (production is untouched) =========================="
EXTRA_ARGS=(--extra-data "$CORRECTIONS")
if [ "$USE_PSEUDO" -eq 1 ] && [ -f "$PSEUDO" ]; then
  EXTRA_ARGS+=(--extra-data "$PSEUDO")
fi
run "$PY" train_arabert.py \
  --model-name model_output/bert_adapted \
  --output-dir "$CANDIDATE_DIR" \
  "${EXTRA_ARGS[@]}"

echo "=== 5/6  gate: candidate vs active, on the GOLD set ==========================="
run "$PY" eval/eval_classifier.py \
  --compare "$ACTIVE_DIR" "$CANDIDATE_DIR" \
  --gold "$GOLD_DIR" --out "$REPORT"

echo "=== 6/6  activate only on an improvement ======================================"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  (dry run: no activation)"
  exit 0
fi
"$PY" - "$REPORT" "$CANDIDATE_DIR" <<'PYGATE'
import json, sys
from pathlib import Path

sys.path.insert(0, ".")
from app.db import init_db, session_scope
from app.db.repo import register_model

report_path, candidate_dir = sys.argv[1], sys.argv[2]
results = json.loads(Path(report_path).read_text(encoding="utf-8"))["results"]
active, candidate = results[0], results[1]

def score(entry):
    # The gold set decides; the synthetic split is only a sanity check, because every
    # model scores well on the distribution it was trained on.
    gold = entry.get("gold_real")
    return (gold or entry.get("synthetic_test") or {}).get("macro_f1", 0.0), bool(gold)

active_f1, on_gold = score(active)
candidate_f1, _ = score(candidate)
delta = (candidate_f1 - active_f1) * 100
basis = "gold set" if on_gold else "SYNTHETIC split (no gold set yet — weak evidence)"

print(f"  active   : {active_f1:.4f}")
print(f"  candidate: {candidate_f1:.4f}   ({delta:+.2f} points, on the {basis})")

if candidate_f1 <= active_f1:
    print("  -> NOT activated: the candidate is not better. Production is unchanged.")
    print("     If pseudo-labels were included, re-run with --no-pseudo to test whether "
          "they caused the regression.")
    raise SystemExit(1)

init_db()
with session_scope() as session:
    register_model(session, kind="classifier", version=Path(candidate_dir).name,
                   path=candidate_dir, metrics={"macro_f1": candidate_f1},
                   activate=True)
print(f"  -> ACTIVATED {candidate_dir} (rollback: flip is_active in model_registry)")
PYGATE

echo
echo "Cycle complete. Report: $REPORT"
