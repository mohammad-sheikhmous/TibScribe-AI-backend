from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(cond: bool, ok: str, fail: str, errors: list[str]) -> None:
    print(("[PASS] " + ok) if cond else ("[FAIL] " + fail))
    if not cond:
        errors.append(fail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-model", action="store_true", help="also require the real model_output checkpoint")
    args = ap.parse_args()
    errors: list[str] = []

    modal_req = (ROOT / "ai-service/requirements.modal.txt").read_text(encoding="utf-8")
    check(not re.search(r"(?m)^arabert(?:[<=> ]|$)", modal_req), "Modal runtime preserves manual preprocessing", "requirements.modal.txt installs arabert", errors)
    check("psycopg[binary]" in modal_req, "AI PostgreSQL driver present", "AI PostgreSQL driver missing", errors)

    db_cfg = (ROOT / "laravel-backend/config/database.php").read_text(encoding="utf-8")
    check("'pgsql'" in db_cfg, "Laravel pgsql connection configured", "Laravel pgsql connection missing", errors)
    docker = (ROOT / "laravel-backend/Dockerfile.render").read_text(encoding="utf-8")
    check("pdo_pgsql" in docker, "Render image includes pdo_pgsql", "Render image missing pdo_pgsql", errors)
    check((ROOT / "render.yaml").is_file(), "Render Blueprint present", "render.yaml missing", errors)
    check((ROOT / "ai-service/modal_app.py").is_file(), "Modal app present", "modal_app.py missing", errors)
    check((ROOT / "deployment/react/TibScribe-AI-FINAL-React.postman_collection.json").is_file(), "React Postman handoff present", "React Postman collection missing", errors)

    # Secrets must never be committed in the release package.
    suspect = []
    for p in ROOT.rglob(".env"):
        suspect.append(str(p.relative_to(ROOT)))
    check(not suspect, "No real .env files committed", "real .env files found: " + ", ".join(suspect), errors)

    if args.require_model:
        model = ROOT / "ai-service/model_output"
        required = ["best_model.pt", "label_mapping.json", "model_config.json", "train_stats.json", "bert", "tokenizer"]
        missing = [x for x in required if not (model / x).exists()]
        check(not missing, "model_output checkpoint complete", "model_output missing: " + ", ".join(missing), errors)
    else:
        print("[INFO] model_output intentionally not required in packaged RC; upload it to Modal Volume before deploy")

    if not (ROOT / "laravel-backend/composer.lock").exists():
        print("[WARN] composer.lock was absent from the supplied V7 source. Functional deploy is supported, but add a lockfile before a strict reproducible production release.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
