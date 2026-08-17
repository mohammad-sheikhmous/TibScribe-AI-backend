param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1,63)]
    [int]$Rule
)

$Report = "EXPERTA_MED\samples\follow4_json_tests\test_d$Rule.json"
$Name = "test_d$Rule"
python -m EXPERTA_MED $Report --name $Name
Write-Host ""
Write-Host "Output:"
Write-Host "EXPERTA_MED\output\$Name.suggestions.json"
