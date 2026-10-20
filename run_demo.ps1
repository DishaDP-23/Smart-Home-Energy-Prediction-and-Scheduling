$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create a Python virtual environment (.venv) and install requirements.txt first. See README.md.'
}
& $projectPython (Join-Path $PSScriptRoot 'generate_sample.py')
if ($LASTEXITCODE -ne 0) { throw 'Sample generation failed' }
& $projectPython (Join-Path $PSScriptRoot 'evaluate.py')
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
