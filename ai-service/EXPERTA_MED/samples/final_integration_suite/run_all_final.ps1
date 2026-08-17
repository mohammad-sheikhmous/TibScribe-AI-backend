$ErrorActionPreference = "Stop"
$Reports = @(
    "final_01_preconception_multirisk"
    "final_02_infertility_initial_crosssource"
    "final_03_female_infertility_diagnostics"
    "final_04_pcos_treatment_escalation"
    "final_05_male_factor_infertility"
    "final_06_unexplained_infertility_pathway"
    "final_07_early_pregnancy_multidomain"
    "final_08_high_risk_pregnancy_emergency"
    "final_09_complicated_active_labour"
    "final_10_second_stage_delivery"
    "final_11_immediate_newborn_care"
    "final_12_routine_immediate_postpartum"
    "final_13_postpartum_emergency"
    "final_14_very_small_newborn_danger"
    "final_15_discharge_followup"
)

foreach ($Report in $Reports) {
    Write-Host "Running $Report ..."
    python -m EXPERTA_MED "EXPERTA_MED\samples\final_integration_suite\$Report.json" --name $Report
}

Write-Host ""
Write-Host "Finished. Results are in EXPERTA_MED\output"
