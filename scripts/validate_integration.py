"""Static/runtime-light validation of the Laravel <-> FastAPI integration contract.

This script deliberately does not import app.main (which would load the ML runtime).
It mounts the individual FastAPI routers in a probe application and compares their
real method/path table with the paths the Laravel gateway calls.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
errors: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


# --- Laravel source integrity -------------------------------------------------------
php_files = list((root / "laravel-backend").rglob("*.php"))
for file in php_files:
    result = subprocess.run(["php", "-l", str(file)], capture_output=True, text=True)
    check(result.returncode == 0, f"PHP syntax: {file}: {result.stdout + result.stderr}")

bootstrap = (root / "laravel-backend/bootstrap/app.php").read_text()
check("api: __DIR__.'/../routes/api.php'" in bootstrap,
      "Laravel routes/api.php is not registered through withRouting(api: ...)")
check("group(base_path('routes/api.php'))" not in bootstrap,
      "Duplicate routes/api.php registration appears to have returned")

routes = (root / "laravel-backend/routes/doctor-api.php").read_text()
service = (root / "laravel-backend/app/Services/AiMedicalService.php").read_text()
normalized_routes = routes.replace(" ", "")
check("['auth:doctor','doctor-token','throttle:normal-apis']" in normalized_routes,
      "Laravel clinical/account routes are not restricted to full doctor tokens")
check("->post('passwords/reset'" in routes and "['auth:doctor', 'throttle:normal-apis']" in routes,
      "Password-reset route is not isolated from full doctor-token routes")
check(all(header in service for header in (
    "X-TibScribe-Doctor-ID", "X-TibScribe-Patient-ID", "X-TibScribe-Session-ID"
)), "Laravel does not forward trusted correlation headers")
check("sessions/{session}/suggestions/{suggestion}/feedback" in routes,
      "Suggestion feedback is not session-scoped")
check("$data['actor'] = 'doctor:'" in (root / "laravel-backend/app/Http/Controllers/Doctor/ClinicalArtifactController.php").read_text(),
      "Correction audit actor can be supplied by the browser")
doctor_token_mw = (root / "laravel-backend/app/Http/Middleware/EnsureDoctorToken.php").read_text()
check("$doctor->tokenCan('*')" in doctor_token_mw,
      "Password-reset tokens can leak into general medical APIs")
check("client(false)->patch" in service and "client(false)->post" in service,
      "Non-idempotent AI writes are using automatic HTTP retries")
check("Idempotency-Key" in (root / "laravel-backend/app/Http/Controllers/Doctor/ClinicalSessionController.php").read_text(),
      "Laravel public session creation has no client idempotency support")
check("['required', 'uuid']" in (root / "laravel-backend/app/Http/Controllers/Doctor/ClinicalSessionController.php").read_text(),
      "Laravel public clinical-session idempotency key is not mandatory")
check("client_request_id" in (root / "laravel-backend/app/Models/ClinicalSession.php").read_text(),
      "ClinicalSession does not persist the public idempotency key")
check("'client_request_fingerprint' => $requestFingerprint" in (root / "laravel-backend/app/Http/Controllers/Doctor/ClinicalSessionController.php").read_text(),
      "Laravel does not persist the original audio fingerprint for retry recovery")
check("Retry audio does not match the original clinical-session recording." in (root / "laravel-backend/app/Http/Controllers/Doctor/ClinicalSessionController.php").read_text(),
      "Laravel retry path does not bind re-uploaded audio to the original session fingerprint")
check("clinical-session-mutation:" in (root / "laravel-backend/app/Http/Controllers/Doctor/PatientController.php").read_text(),
      "Session-linked PatientState/KBS refresh is not serialized with report mutations")

# --- Real FastAPI router table without ML startup ----------------------------------
sys.path.insert(0, str(root / "ai-service"))
from app.api.audio import router as audio_router  # noqa: E402
from app.api.corrections import router as corrections_router  # noqa: E402
from app.api.jobs import router as jobs_router  # noqa: E402
from app.api.patients import router as patients_router  # noqa: E402
from app.api.suggestions import router as suggestions_router  # noqa: E402

routers = (
    jobs_router,
    audio_router,
    corrections_router,
    patients_router,
    suggestions_router,
)

actual = {
    (method, route.path)
    for router in routers
    for route in router.routes
    for method in (getattr(route, "methods", None) or set())
    if method not in {"HEAD", "OPTIONS"}
}

expected = {
    ("POST", "/jobs"),
    ("POST", "/jobs/{job_id}/retry"),
    ("GET", "/jobs/{job_id}"),
    ("GET", "/jobs/{job_id}/report"),
    ("GET", "/jobs/{job_id}/transcript"),
    ("GET", "/jobs/{job_id}/suggestions"),
    ("GET", "/jobs/{job_id}/corrections"),
    ("GET", "/jobs/{job_id}/review-queue"),
    ("PATCH", "/jobs/{job_id}/items/{item_id}"),
    ("GET", "/jobs/{job_id}/audio"),
    ("GET", "/jobs/{job_id}/items/{item_id}/audio"),
    ("POST", "/suggestions/{suggestion_id}/feedback"),
    ("PUT", "/patients/by-external/{source}/{external_id}"),
    ("GET", "/patients/{patient_id}/timeline"),
    ("POST", "/patients/{patient_id}/state"),
}
for endpoint in sorted(expected):
    check(endpoint in actual, f"FastAPI gateway contract missing {endpoint[0]} {endpoint[1]}")

# Laravel proxy implementation must call every internal path family it depends on.
for needle in (
    "->post('/jobs'", "'/retry'", "'/report'", "'/transcript'", "'/suggestions'",
    "'/corrections'", "'/review-queue'", "'/items/'", "'/feedback'",
    "->put(\n            '/patients/by-external/laravel/", "'/timeline'", "'/state'", "'/audio'",
):
    check(needle in service, f"Laravel AI proxy no longer contains expected call fragment: {needle}")

# --- FastAPI shared-secret boundary -------------------------------------------------
main = (root / "ai-service/app/main.py").read_text()
deps = (root / "ai-service/app/api/deps.py").read_text()
check("dependencies=_service_auth" in main, "FastAPI business routers are not protected by service auth")
check("hmac.compare_digest" in deps, "Service token comparison is not constant-time")
jobs_source = (root / "ai-service/app/api/jobs.py").read_text()
check("Clinical-session idempotency key was replayed with different audio." in jobs_source,
      "FastAPI idempotency key is not bound to the original audio SHA-256")
check("Patient + Visit + Job are one DB transaction" in jobs_source,
      "FastAPI gateway upload no longer guarantees atomic Visit+Job creation")

# --- Reproducible/runtime deployment guards -----------------------------------------
ai_docker = (root / "ai-service/Dockerfile").read_text()
requirements = (root / "ai-service/requirements.lock").read_text()
ci_path = root / ".github/workflows/ci.yml"
# Deployment ZIPs intentionally may omit repository-only GitHub metadata.  Validate
# CI drift when the workflow is present, but do not make the runtime release artifact
# unverifiable just because .github/ was excluded from packaging.
ci = ci_path.read_text() if ci_path.exists() else ""
laravel_docker = (root / "laravel-backend/Dockerfile").read_text()
compose = (root / "docker-compose.yml").read_text()
check("FROM python:3.12" in ai_docker,
      "AI Docker runtime must be Python 3.12 because scipy==1.18.0 requires >=3.12")
check("python scripts/migrate_db.py" in ai_docker,
      "AI Docker startup does not upgrade Alembic/legacy databases")
check("fastapi==0.139.2" in requirements,
      "FastAPI should include the 0.139.2 router concurrency fix")
check("torch==2.13.0 is installed explicitly" in requirements and "\ntorch==2.13.0\n" not in requirements,
      "Torch CPU wheel is duplicated in requirements.lock")
if ci:
    check(ci.count("python-version: '3.12'") >= 2,
          "CI Python version has drifted from the Python 3.12 runtime")
check("mbstring" in laravel_docker and "upload_max_filesize=105M" in laravel_docker,
      "Laravel image is missing required mbstring or the 100 MB audio upload PHP limits")
check("composer install --no-dev" in laravel_docker,
      "Laravel production image is installing development dependencies")
laravel_block = compose.split("  laravel:", 1)[1].split("  laravel-queue:", 1)[0]
check("ai-service:" not in laravel_block.split("depends_on:", 1)[-1] if "depends_on:" in laravel_block else True,
      "Laravel availability is incorrectly coupled to AI readiness")
check("--timeout=60" in compose, "Queue timeout must remain below database retry_after=90")
check("python scripts/migrate_db.py" in ai_docker, "AI migration bootstrap is missing")
check("client_request_id" in "".join(
    f.read_text() for f in (root / "laravel-backend/database/migrations").glob("*.php")
), "Laravel session idempotency migration is missing")

# --- Canonical AI/KBS build contract ------------------------------------------------
result = subprocess.run(
    [sys.executable, "scripts/validate_final_build.py", "--source-only"],
    cwd=root / "ai-service", capture_output=True, text=True,
)
check(result.returncode == 0, "AI final-build source validation failed:\n" + result.stdout + result.stderr)

print(f"Validated {len(php_files)} PHP files + {len(expected)} Laravel/FastAPI gateway endpoints.")
if errors:
    for error in errors:
        print("[FAIL]", error)
    raise SystemExit(1)
print("[PASS] Cross-service source integration contract.")
