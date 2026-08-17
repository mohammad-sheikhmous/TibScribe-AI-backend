param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1,28)]
    [int]$Rule
)

$Report = "EXPERTA_MED\samples\follow2_json_tests\test_n$Rule.json"
$Name = "test_n$Rule"

python -m EXPERTA_MED $Report --name $Name
Write-Host ""
Write-Host "Output:"
Write-Host "EXPERTA_MED\output\$Name.suggestions.json"
