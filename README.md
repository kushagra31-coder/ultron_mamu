# Ultron Dynamic Local Voice Agent

Dynamic local Windows voice agent built around:

- Whisper/faster-whisper for local speech-to-text
- Ollama + Qwen2.5-Coder 7B for local reasoning and tool selection
- Ollama native tool calling through the official Python SDK
- Automatic tool discovery from `tools/*.py`
- Offline Windows TTS through `pyttsx3`/SAPI
- `ddgs` for live web/YouTube search without a separate search API key

## Important Windows prerequisite

This project intentionally uses **64-bit Python 3.11**. The pinned NumPy 1.26.4 dependency has an official Windows CPython 3.11 wheel. Using Python 3.14 causes pip to fall back to a source build for this pinned NumPy version, which requires a C/C++ compiler.

The installer therefore creates the project venv with `py -3.11` and recreates an existing venv if it was made with another Python version.

## Install

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

The setup script will:

1. find Python 3.11 through the Windows Python Launcher;
2. create/recreate `venv` with Python 3.11;
3. install only binary wheels from `requirements.txt`;
4. verify Ollama is reachable;
5. verify `qwen2.5-coder:7b` exists;
6. verify Python packages and dynamic tool discovery.

## Run

Terminal 1:

```powershell
.\run_agent.ps1
```

Terminal 2:

```powershell
.\run_stt.ps1
```

Hotkey: `Ctrl+Alt+Space`

## Add a capability

Create another `.py` module under `tools/`, import the `tool` decorator, and decorate functions with `@tool(...)`.

The core `agent_server.py` does not need to be edited.

## Repair an old/wrong venv manually

If an older setup already created `venv` with Python 3.14:

```powershell
Remove-Item -Recurse -Force .\venv
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
python .\verify_install.py
```
