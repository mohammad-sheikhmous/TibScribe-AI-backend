#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# EXPERTA_MED — server-side setup (run once after unpacking the bundle).
#
#   bash setup_server.sh
#
# Builds .venv, installs dependencies, registers a Jupyter kernel, and reports
# whether the box can actually train (GPU) or only limp along on CPU.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"
echo "project root: $ROOT"

# --- 1. venv ---------------------------------------------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv 2>/dev/null || {
    echo "❌ python3 -m venv failed. Install it first:  sudo apt install -y python3-venv"
    exit 1
  }
fi
. .venv/bin/activate
python -m pip install -qU pip wheel setuptools

# --- 2. dependencies -------------------------------------------------------
# Core first: everything training and inference actually need. Whisper/ASR is
# installed separately so a missing system ffmpeg cannot block the classifier.
echo "--- installing core ML deps (this is the slow part) ---"
pip install -q torch transformers scikit-learn numpy datasets pyyaml || {
  echo "❌ core install failed"; exit 1; }

echo "--- installing notebook tooling ---"
pip install -q jupyterlab ipykernel matplotlib

echo "--- installing the rest of requirements.txt (failures here are non-fatal) ---"
pip install -q -r requirements.txt || \
  echo "⚠️  some optional deps failed (likely openai-whisper needing system ffmpeg) — training/inference are unaffected"

# --- 3. Jupyter kernel -----------------------------------------------------
python -m ipykernel install --user \
  --name experta-med --display-name "Python (EXPERTA_MED)" >/dev/null 2>&1 \
  && echo "✅ kernel registered: Python (EXPERTA_MED)" \
  || echo "⚠️  kernel registration failed — select .venv/bin/python manually in Jupyter"

# --- 4. what did we land on? ----------------------------------------------
echo
echo "=================== environment ==================="
python - <<'PY'
import platform, sys
print("python      :", sys.version.split()[0], "|", platform.platform())
try:
    import torch
    print("torch       :", torch.__version__)
    print("cuda        :", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu[{i}]   : {p.name}  {p.total_memory/1024**3:.1f} GB")
    else:
        import os
        print("  ⚠️  CPU only — full training (20 epochs) will take many hours.")
        print("  cpu cores :", os.cpu_count())
except Exception as e:
    print("torch       : NOT IMPORTABLE —", e)
PY

echo
command -v ffmpeg >/dev/null && echo "ffmpeg      : $(command -v ffmpeg)" \
  || echo "ffmpeg      : MISSING (only needed for Whisper/audio, not for training)"

echo
echo "data.jsonl  : $([ -f data.jsonl ] && wc -l < data.jsonl || echo MISSING) lines"
echo "checkpoint  : $([ -f model_output/best_model.pt ] && echo present || echo 'absent — train first, or upload best_model.pt')"

echo
echo "=================== next ==================="
echo "1. In Jupyter open:  EXPERTA_MED/notebooks/EXPERTA_MED_train_infer.ipynb"
echo "2. Kernel → Change kernel → Python (EXPERTA_MED)"
echo "3. Run sections 1-3, then 4 to train."
echo
echo "Or train straight from this terminal:"
echo "   cd '$ROOT' && .venv/bin/python train_arabert.py --epochs 20 --batch-size 16"
