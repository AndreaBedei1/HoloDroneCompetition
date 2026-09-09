param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$RunDir
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
conda run -n marine_race_rl python -m marine_race_arena.learning.longrun_tools status "$RunDir"
exit $LASTEXITCODE
