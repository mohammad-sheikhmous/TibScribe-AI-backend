$Rules = 1..48
foreach ($Rule in $Rules) {
    $Name = "test_r$Rule"
    Write-Host "Running $Name ..."
    python -m EXPERTA_MED "EXPERTA_MED\samples\follow5_raw_tests_active\$Name.json" --name $Name
}
Write-Host "Finished. Results are in EXPERTA_MED\output"
