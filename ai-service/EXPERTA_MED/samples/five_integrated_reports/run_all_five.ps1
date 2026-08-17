$Reports = @(
    "integrated_1_early_pregnancy",
    "integrated_2_high_risk_pregnancy",
    "integrated_3_complicated_labour",
    "integrated_4_normal_delivery_newborn",
    "integrated_5_postpartum_discharge"
)

foreach ($Report in $Reports) {
    Write-Host "Running $Report ..."
    python -m EXPERTA_MED "EXPERTA_MED\samples\five_integrated_reports\$Report.json" --name $Report
}

Write-Host ""
Write-Host "Finished. Results are in EXPERTA_MED\output"
