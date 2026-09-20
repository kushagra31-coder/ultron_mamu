$ErrorActionPreference="Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
& ".\venv\Scripts\Activate.ps1"
python -m uvicorn agent_server:app --host 127.0.0.1 --port 8000 --log-level warning
