param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1,50)]
    [int]$Rule
)

$Report = "EXPERTA_MED\samples\follow1_json_tests\test_a$Rule.json"
$Name = "test_a$Rule"

python -m EXPERTA_MED $Report --name $Name
Write-Host ""
Write-Host "Output:"
Write-Host "EXPERTA_MED\output\$Name.suggestions.json"
