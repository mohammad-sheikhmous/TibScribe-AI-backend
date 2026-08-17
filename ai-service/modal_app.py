"""Modal GPU deployment for TibScribe AI.

Run from ai-service/:
  pip install "modal>=1.1.4,<2"
  modal setup
  modal volume create tibscribe-models
  modal volume create tibscribe-ai-data
  modal volume create tibscribe-cache
  modal volume put tibscribe-models ./model_output /model_output
  modal secret create tibscribe-ai-secrets --from-dotenv .env.modal
  modal run modal_app.py::validate_model_volume
  modal run modal_app.py::warm_whisper
  modal deploy modal_app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "tibscribe-ai"
AI_ROOT = Path(__file__).resolve().parent
REMOTE_APP = "/app"
MODEL_MOUNT = "/models"
DATA_MOUNT = "/data"
CACHE_MOUNT = "/mnt/tibscribe-cache"

model_volume = modal.Volume.from_name("tibscribe-models", create_if_missing=True)
data_volume = modal.Volume.from_name("tibscribe-ai-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("tibscribe-cache", create_if_missing=True)
secrets = modal.Secret.from_name("tibscribe-ai-secrets")

# Reproduce the known-good Kaggle preprocessing/runtime. `arabert` is intentionally absent.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "build-essential")
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools==78.1.0 wheel",
        "python -m pip install --no-build-isolation frozendict==1.2 schema==0.6.7",
    )
    .pip_install("torch==2.13.0")
    .pip_install_from_requirements(str(AI_ROOT / "requirements.modal.txt"))
    .env(
        {
            "PYTHONPATH": REMOTE_APP,
            "MODEL_DIR": f"{MODEL_MOUNT}/model_output",
            "WHISPER_MODEL_SIZE": "large-v3",
            "ASR_LANGUAGE": "ar",
            "STRICT_MODEL_CHECKS": "true",
            "SERVICE_AUTH_REQUIRED": "true",
            "GATEWAY_IDENTITY_REQUIRED": "true",
            "SERVICE_NAME": "laravel",
            "UPLOAD_DIR": f"{DATA_MOUNT}/uploads",
            "RESULT_DIR": f"{DATA_MOUNT}/results",
            "AUDIO_DIR": f"{DATA_MOUNT}/audio",
            # One GPU pipeline at a time; HTTP polling remains concurrent via modal.concurrent.
            "EXECUTOR_MAX_WORKERS": "1",
            "MAX_CONCURRENT_JOBS": "4",
            "MAX_UPLOAD_MB": "100",
        }
    )
    .add_local_dir(
        str(AI_ROOT),
        remote_path=REMOTE_APP,
        ignore=[
            "model_output/**",
            "data/**",
            "tests/**",
            "eval/**",
            "notebooks/**",
            "reports/**",
            ".pytest_cache/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".env*",
        ],
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={MODEL_MOUNT: model_volume},
    timeout=120,
    memory=2048,
)
def validate_model_volume() -> dict[str, object]:
    """Cheap deployment guard before allocating a GPU container."""
    root = Path(MODEL_MOUNT) / "model_output"
    required = [
        "best_model.pt",
        "label_mapping.json",
        "model_config.json",
        "train_stats.json",
        "bert",
        "tokenizer",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"model_output incomplete; missing: {', '.join(missing)}")

    config = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    preprocessing = config.get("preprocessing")
    if preprocessing != "manual":
        raise RuntimeError(
            f"Expected trained preprocessing='manual', got {preprocessing!r}. "
            "Do not change the serving preprocessor silently."
        )
    return {
        "status": "ok",
        "model_dir": str(root),
        "preprocessing": preprocessing,
        "model_name": config.get("model_name"),
        "num_classes": config.get("num_classes"),
    }


@app.function(
    image=image,
    volumes={CACHE_MOUNT: cache_volume},
    timeout=1800,
    memory=8192,
)
def warm_whisper() -> dict[str, str]:
    """Download Whisper large-v3 once into the persistent cache Volume."""
    import os

    os.environ["XDG_CACHE_HOME"] = CACHE_MOUNT
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/huggingface"

    import whisper

    cache_dir = Path(CACHE_MOUNT) / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(
        "large-v3",
        device="cpu",
        download_root=str(cache_dir),
    )

    del model
    cache_volume.commit()

    return {
        "status": "ok",
        "cache": str(cache_dir),
    }

@app.function(
    image=image,
    gpu="L4",
    memory=16384,
    timeout=1200,
    startup_timeout=600,
    max_containers=1,
    scaledown_window=300,
    secrets=[secrets],
    volumes={
        MODEL_MOUNT: model_volume,
        DATA_MOUNT: data_volume,
        CACHE_MOUNT: cache_volume,
    },
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    """Serve the existing FastAPI application without exposing model files in the image."""

    os.environ["XDG_CACHE_HOME"] = CACHE_MOUNT
    os.environ["HF_HOME"] = f"{CACHE_MOUNT}/huggingface"

    os.chdir(REMOTE_APP)

    # Alembic is idempotent. DATABASE_URL / optional MIGRATION_DATABASE_URL come from Secret.
    subprocess.run([sys.executable, "scripts/migrate_db.py"], check=True)

    from app.main import app as web_app

    # runner.py calls this after every success/failure so audio/results survive scale-down.
    web_app.state.storage_commit_hook = data_volume.commit
    return web_app
