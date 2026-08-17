param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1,41)]
    [int]$Rule
)

$Report = "EXPERTA_MED\samples\follow3_json_tests\test_l$Rule.json"
$Name = "test_l$Rule"
python -m EXPERTA_MED $Report --name $Name
Write-Host ""
Write-Host "Output: EXPERTA_MED\output\$Name.suggestions.json"
