$ErrorActionPreference = "Stop"
for ($i = 1; $i -le 140; $i++) {
    $name = "test_r$i"
    Write-Host "Running $name ..."
    python -m EXPERTA_MED "EXPERTA_MED\samples\follow7_raw_tests_R1_R140\$name.json" --name $name
}
Write-Host "Finished. Results are in EXPERTA_MED\output"
