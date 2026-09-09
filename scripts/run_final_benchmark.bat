@echo off
REM One-command final common benchmark.
REM
REM Runs every controller under test (PPO 525678 / 900462 / 1000814, BC-v3, the
REM deterministic rule baseline and the hybrid baseline) through the identical
REM suite -- single-gate retention, two-gate straight/left/right, vertical
REM low-to-high and high-to-low transitions, three-gate sequences, S-shapes and
REM the three official current-free circuits -- on the same seeds, the same
REM geometries and the same real-HoloOcean configuration, then aggregates,
REM compares, plots and writes the Markdown report.
REM
REM   run_final_benchmark.bat [OUT_DIR] [WORKERS]
REM
REM Safe to re-run: finished episodes are skipped, so an interrupted benchmark
REM resumes where it stopped.
setlocal
cd /d "%~dp0\.."

set OUT=%~1
if "%OUT%"=="" set OUT=results/rl/final_benchmark/reliability_first_20260731
set WORKERS=%~2
if "%WORKERS%"=="" set WORKERS=6

echo [final-benchmark] output   : %OUT%
echo [final-benchmark] workers  : %WORKERS%
echo [final-benchmark] adapter  : holoocean (no fallback), currents disabled

conda run -n marine_race_rl python -m marine_race_arena.learning.final_benchmark run ^
  --out "%OUT%" --workers %WORKERS% ^
  --episodes-per-case 8 --official-episodes 5 ^
  --adapter holoocean --current-profile none ^
  --video --video-stride 5 --retries 3 --repair
set RC=%ERRORLEVEL%
if not "%RC%"=="0" echo [final-benchmark] WARNING: a shard exited with %RC%; the report still covers every completed episode.

conda run -n marine_race_rl python -m marine_race_arena.learning.final_benchmark_report --out "%OUT%"
set RC_REPORT=%ERRORLEVEL%

echo.
echo ==================== FINAL BENCHMARK ====================
echo   episodes  : %OUT%/episodes.json  (+ episodes.csv)
echo   aggregate : %OUT%/aggregate_by_group.csv
echo   paired    : %OUT%/paired_comparisons.csv
echo   report    : %OUT%/final_benchmark_report.md
echo   plots     : %OUT%/plots
echo   videos    : %OUT%/artifacts
echo =========================================================
exit /b %RC_REPORT%
