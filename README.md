# Ultron — Local Windows Voice Agent

A voice-controlled Windows assistant that runs almost entirely on your own hardware. Speak a command, it plans and executes real actions (opening apps, typing, reading Calculator's display, web search, file ops), and talks back.

## Architecture

```
Mic → faster-whisper (local STT)
  → [optional] Groq intent router — "chat" or "action"?
      ├─ chat → answered directly by Groq
      └─ action → Qwen3.5:4b (local, Ollama) — structured-output planner
          → executes tools, loops until done
          → spoken reply (Groq if enabled, else local)
          → pyttsx3/SAPI (local TTS)
```

Everything works fully offline by default. Groq is strictly opt-in (`ULTRON_TALK_BACKEND=groq`) — see [Privacy](#privacy) below for exactly what that changes.

## How tool calling actually works

This does not rely on Ollama's native `tool_calls` mechanism — that turned out to be unreliable across Qwen model versions and Ollama releases (narration instead of calls, silently dropped calls, truncated output). Instead:

- Every tool function in `tools/*.py` is introspected at request time (`build_tool_context()`), and a Pydantic model is built dynamically from the actual function signatures — tool names become a Literal[...] enum, so a hallucinated tool name is rejected at validation time, not silently passed through.
- The model's output is grammar-constrained via Ollama's `format=<json schema>` — the model is mechanically unable to produce prose instead of a structured call, because every output token is masked to fit the schema.
- The planner runs in a loop (up to `ULTRON_MAX_STEPS`, default 30): plan → execute → feed result back → replan, with duplicate-action detection so a confused model can't retype the same text forever.

## Install

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

This project intentionally uses **64-bit Python 3.11** — the pinned NumPy build has an official Windows wheel for 3.11 but not for newer interpreters, which otherwise forces pip into a source build requiring a C/C++ compiler. The installer creates the venv with `py -3.11` and recreates it if an existing venv was built with the wrong version.

The script also:
- Verifies Ollama is reachable
- Verifies `qwen3.5:4b` (the default `ULTRON_MODEL`) is pulled
- Verifies Python packages and dynamic tool discovery

## Run

**Terminal 1:**
```powershell
.\run_agent.ps1
```

**Terminal 2:**
```powershell
.\run_stt.ps1
```

**Hotkey:** `Ctrl+Alt+Space`

## Add a capability

Create another `.py` module under `tools/`, decorate a function with the `@tool(...)` registry decorator. `agent_server.py` does not need to be edited — `build_tool_context()` discovers it automatically from its signature and docstring on the next request.

## Optional: Groq backend

Set both to enable:
```powershell
$env:GROQ_API_KEY="your-key"
$env:ULTRON_TALK_BACKEND="groq"
```

Leave `ULTRON_TALK_BACKEND` unset (default `local`) and none of this activates — the planner always runs locally regardless of this setting.

### What this actually changes

- **`GROQ_CLASSIFIER_MODEL` (default `groq/compound-mini`):** every user utterance is sent here first, to classify it as chat or action, before any local processing happens.
- **chat-classified requests** are answered entirely by Groq (`GROQ_MODEL`, default `openai/gpt-oss-20b`) — the local model is never invoked for these.
- **action-classified requests** run the full local planner as normal; only the final one-sentence spoken confirmation goes through Groq instead of the local model, if enabled.
- Classification defaults to action (full local pipeline) on any failure — bad key, network error, rate limit — so a request is never silently dropped.

## Privacy

If `ULTRON_TALK_BACKEND=groq` is set, the raw text of every single utterance leaves your machine for classification, regardless of whether it ends up being handled locally or not. This is printed as a startup warning by the server itself, not just documented here. Set it to `local` (or leave it unset) to keep 100% of input on-device — nothing about tool execution, file access, or GUI control ever depends on Groq being enabled.

## Known limits worth knowing before you rely on this

- **No wall-clock timeout on the planner loop** — only a step count (`ULTRON_MAX_STEPS`). A confused model taking 30 slightly-different-but-still-wrong steps could take a long time before giving up.
- **GUI automation is a mix of deterministic UIAutomation** (e.g. `read_calculator_display`) where a dedicated tool exists, and vision/screenshot-based `capture_screen` as the general fallback elsewhere — the former is faster and more reliable; prefer building dedicated UIAutomation tools for any app you use often.
- **Tool control (file access, app launching, GUI input) is Windows-specific throughout** (`pywinauto`) — not portable to macOS/Linux without rewriting that layer.
