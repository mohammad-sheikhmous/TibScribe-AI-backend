$ErrorActionPreference = "Stop"

cd C:\Users\enter\Python

python EXPERTA_MED\samples\final_integration_suite\validate_input_schema.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python EXPERTA_MED\samples\final_integration_suite\inspect_final_extraction.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\EXPERTA_MED\samples\final_integration_suite\run_all_final.ps1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python EXPERTA_MED\samples\final_integration_suite\check_final_results.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
