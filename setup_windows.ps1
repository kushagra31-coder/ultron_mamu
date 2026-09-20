$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=== ULTRON Windows setup ===" -ForegroundColor Cyan

if ($env:OS -ne "Windows_NT") {
    throw "This installer is Windows-only."
}

# We deliberately use Python 3.11 because this project pins NumPy 1.26.4
# and faster-whisper 1.0.3. NumPy 1.26.4 has an official Windows CPython 3.11
# wheel; Python 3.14 falls back to a source build and needs a C/C++ compiler.
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pyLauncher) {
    throw "Python Launcher 'py.exe' was not found. Install 64-bit Python 3.11 for Windows, then rerun this script."
}

$py311 = $null
try {
    $py311 = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim()
} catch {
    $py311 = $null
}

if (-not $py311) {
    throw "Python 3.11 was not found. Install 64-bit Python 3.11 for Windows (with the Python Launcher), then rerun this script."
}

Write-Host "[OK] Python 3.11: $py311" -ForegroundColor Green

$venvPython = Join-Path $PWD "venv\Scripts\python.exe"
$recreateVenv = $false

if (Test-Path $venvPython) {
    $venvVersion = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($venvVersion -ne "3.11") {
        Write-Host "[INFO] Existing venv uses Python $venvVersion; recreating it with Python 3.11." -ForegroundColor Yellow
        $recreateVenv = $true
    }
}

if ($recreateVenv -and (Test-Path ".\venv")) {
    Remove-Item -Recurse -Force ".\venv"
}

if (-not (Test-Path $venvPython)) {
    & py -3.11 -m venv venv
}

& $venvPython -c "import sys; if False: pass" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "The Python 3.11 virtual environment could not be created."
}

Write-Host "[OK] Virtual environment: $venvPython" -ForegroundColor Green

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --only-binary=:all: -r .\requirements.txt

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed. No compiler should be needed for the pinned Python 3.11 wheels."
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama was not found on PATH. Install Ollama for Windows first."
}

try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 10
} catch {
    throw "Ollama is installed but not reachable on 127.0.0.1:11434. Start Ollama and rerun this script."
}

$model = "qwen2.5-coder:7b"
$models = & ollama list
if (-not ($models -match [regex]::Escape($model))) {
    Write-Host "[INFO] Pulling $model ..." -ForegroundColor Yellow
    & ollama pull $model
    if ($LASTEXITCODE -ne 0) {
        throw "Ollama could not pull $model."
    }
}

Write-Host "[INFO] Running local verification..." -ForegroundColor Cyan
& $venvPython .\verify_install.py
if ($LASTEXITCODE -ne 0) {
    throw "Verification failed. Fix the reported FAIL items before starting Ultron."
}

Write-Host "" 
Write-Host "Setup and verification completed successfully." -ForegroundColor Green
Write-Host "Start agent: .\run_agent.ps1" -ForegroundColor Green
Write-Host "Start STT:   .\run_stt.ps1" -ForegroundColor Green
