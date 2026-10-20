param([switch]$SkipTraining)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Create .venv and install requirements.txt first.' }
$runArgs = @((Join-Path $PSScriptRoot 'run_refit.py'))
if ($SkipTraining) { $runArgs += '--skip-training' }
& $projectPython @runArgs
if ($LASTEXITCODE -ne 0) { throw 'REFIT pipeline failed' }
