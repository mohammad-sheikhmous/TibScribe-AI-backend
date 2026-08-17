$Reports = @(
    "raw_1_preeclampsia_drug_safety",
    "raw_2_early_pregnancy_nutrition",
    "raw_3_complicated_active_labour",
    "raw_4_second_stage_delivery",
    "raw_5_high_risk_postpartum"
)

foreach ($Report in $Reports) {
    Write-Host "Running $Report ..."
    python -m EXPERTA_MED "EXPERTA_MED\samples\raw_pipeline_tests_no_medical_record\$Report.json" --name $Report
}

Write-Host ""
Write-Host "Finished. Results are in EXPERTA_MED\output"
