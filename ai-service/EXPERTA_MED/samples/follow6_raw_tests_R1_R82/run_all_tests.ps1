$ErrorActionPreference = "Stop"
for ($i = 1; $i -le 82; $i++) {
    $name = "test_r$i"
    Write-Host "Running $name ..."
    python -m EXPERTA_MED "EXPERTA_MED\samples\follow6_raw_tests_R1_R82\$name.json" --name $name
}
Write-Host "Finished. Results are in EXPERTA_MED\output"
