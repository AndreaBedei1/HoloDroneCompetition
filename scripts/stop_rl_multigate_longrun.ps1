param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$RunDir,
    [int]$TimeoutSeconds = 300,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
conda run -n marine_race_rl python -m marine_race_arena.learning.longrun_tools stop "$RunDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$pidFile = Join-Path $RunDir "pid.txt"
$trainingPid = 0
if (Test-Path $pidFile) {
    $trainingPid = [int](Get-Content -Raw -LiteralPath $pidFile).Trim()
}
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ($trainingPid -gt 0 -and (Get-Process -Id $trainingPid -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
}
$alive = $trainingPid -gt 0 -and (Get-Process -Id $trainingPid -ErrorAction SilentlyContinue)
if ($alive -and $Force) {
    Write-Warning "Graceful stop timed out; force-stopping explicitly requested PID $trainingPid."
    Stop-Process -Id $trainingPid -Force
    exit 0
}
if ($alive) {
    Write-Warning "Graceful stop timed out. The process was not killed. Re-run with -Force only after inspecting status."
    exit 3
}
Write-Output "Training stopped after a graceful checkpoint request."
exit 0
