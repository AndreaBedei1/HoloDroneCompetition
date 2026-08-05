param(
    [Parameter(Mandatory=$true)][string]$PpoWorktree,
    [Parameter(Mandatory=$true)][string]$PpoRun,
    [Parameter(Mandatory=$true)][string]$PpoConfig,
    [Parameter(Mandatory=$true)][string]$SacWorktree,
    [Parameter(Mandatory=$true)][string]$SacRun,
    [Parameter(Mandatory=$true)][string]$SacConfig,
    [int]$SacWorkers = 6,
    [string]$StateDir = "results\rl\autonomous_supervisor",
    [string]$ComparisonOutput = "results\rl_public\universal_transition_ppo_vs_sac"
)

$ErrorActionPreference = "Stop"
conda run -n marine_race_rl --no-capture-output python -m marine_race_arena.learning.rl_autonomous_supervisor start `
    --state-dir $StateDir `
    --supervisor-worktree $SacWorktree `
    --comparison-output $ComparisonOutput `
    --ppo-worktree $PpoWorktree --ppo-run $PpoRun --ppo-config $PpoConfig --ppo-workers 2 `
    --sac-worktree $SacWorktree --sac-run $SacRun --sac-config $SacConfig --sac-workers $SacWorkers --sac-allow-fresh
