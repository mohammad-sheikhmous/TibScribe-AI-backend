from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = (
    "best_model.pt",
    "label_mapping.json",
    "model_config.json",
    "train_stats.json",
    "bert",
    "tokenizer",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TibScribe model_output layout.")
    parser.add_argument("model_dir", nargs="?", default="model_output")
    args = parser.parse_args()
    root = Path(args.model_dir)

    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        print("FAIL: missing " + ", ".join(missing))
        return 2

    cfg = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "label_mapping.json").read_text(encoding="utf-8"))
    labels = mapping.get("id2label") or {}
    if cfg.get("preprocessing") != "manual":
        print(f"FAIL: expected preprocessing=manual, got {cfg.get('preprocessing')!r}")
        return 3
    if cfg.get("num_classes") is not None and int(cfg["num_classes"]) != len(labels):
        print(f"FAIL: num_classes={cfg['num_classes']} but mapping has {len(labels)} labels")
        return 4

    print("PASS")
    print(f"model_name={cfg.get('model_name')}")
    print(f"preprocessing={cfg.get('preprocessing')}")
    print(f"num_classes={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
