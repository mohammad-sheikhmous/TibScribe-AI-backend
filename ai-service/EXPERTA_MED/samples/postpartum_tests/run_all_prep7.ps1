# run_all_prep7.ps1
# تشغيل جميع اختبارات Follow7Rules

$ErrorActionPreference = "Stop"
$base = "EXPERTA_MED\samples\postpartum_tests"
$output = "EXPERTA_MED\output"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  تشغيل اختبارات Follow7Rules (140 قاعدة)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$tests = @(
    @{ file = "prep7_01_maternal_assessment.json";  name = "prep7_01" },
    @{ file = "prep7_02_pain_symptoms.json";         name = "prep7_02" },
    @{ file = "prep7_03_preventive_nutrition.json";  name = "prep7_03" },
    @{ file = "prep7_04_mental_health.json";         name = "prep7_04" },
    @{ file = "prep7_05_contraception.json";         name = "prep7_05" },
    @{ file = "prep7_06_newborn_assessment.json";    name = "prep7_06" },
    @{ file = "prep7_07_newborn_preventive.json";    name = "prep7_07" },
    @{ file = "prep7_08_newborn_nutrition.json";     name = "prep7_08" },
    @{ file = "prep7_09_growth_development.json";    name = "prep7_09" },
    @{ file = "prep7_10_discharge_followup.json";    name = "prep7_10" },
    @{ file = "prep7_11_complications.json";         name = "prep7_11" },
    @{ file = "prep7_12_newborn_specific.json";      name = "prep7_12" }
)

$passed = 0
$failed = 0

foreach ($t in $tests) {
    Write-Host ""
    Write-Host "--- $($t.name) ---" -ForegroundColor Yellow
    try {
        python -m EXPERTA_MED "$base\$($t.file)" --name $t.name
        $passed++
        Write-Host "[PASS] $($t.name)" -ForegroundColor Green
    } catch {
        $failed++
        Write-Host "[FAIL] $($t.name): $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  النتيجة: $passed نجح / $failed فشل" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan