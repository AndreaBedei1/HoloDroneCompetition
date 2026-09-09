param(
    [Int64]$TotalTimesteps = 1000000,
    [string]$RunName = "r2_longrun_seed22001",
    [int]$Seed = 22001,
    [string]$CurriculumStage = "C0",
    [string]$ResumeRunDir = "",
    [Int64]$AdditionalTimesteps = 0
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$outputRoot = Join-Path $repo "results\rl\multigate_longrun"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

if ($ResumeRunDir) {
    $runDir = (Resolve-Path $ResumeRunDir).Path
    if ($AdditionalTimesteps -le 0) {
        throw "AdditionalTimesteps must be positive for resume."
    }
    $arguments = @(
        "run", "-n", "marine_race_rl", "python", "-m",
        "marine_race_arena.learning.train_multigate_longrun",
        "--resume", "--run-dir", $runDir,
        "--additional-timesteps", "$AdditionalTimesteps"
    )
    $launcherBase = Join-Path $outputRoot ((Split-Path $runDir -Leaf) + ".resume")
} else {
    $runDir = Join-Path $outputRoot $RunName
    if (Test-Path $runDir) {
        throw "Run directory already exists: $runDir. Use the resume command or a new run name."
    }
    $arguments = @(
        "run", "-n", "marine_race_rl", "python", "-m",
        "marine_race_arena.learning.train_multigate_longrun",
        "--config", "configs/rl_multigate_longrun.json",
        "--total-timesteps", "$TotalTimesteps",
        "--run-name", $RunName,
        "--seed", "$Seed",
        "--curriculum-stage", $CurriculumStage
    )
    $launcherBase = Join-Path $outputRoot ($RunName + ".launcher")
}

$process = Start-Process -FilePath "conda.exe" -ArgumentList $arguments `
    -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput ($launcherBase + ".stdout.log") `
    -RedirectStandardError ($launcherBase + ".stderr.log")

Write-Output "Started PID=$($process.Id)"
Write-Output "Run directory: $runDir"
Write-Output "Status: scripts\status_rl_multigate_longrun.bat `"$runDir`""
Write-Output "Stop:   scripts\stop_rl_multigate_longrun.bat `"$runDir`""
