$ErrorActionPreference="Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
& ".\venv\Scripts\Activate.ps1"
python .\stt_service.py
