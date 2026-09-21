# Ultron — Local Windows Voice Agent

A voice-controlled Windows assistant that runs almost entirely on your own hardware. Speak a command, it plans and executes real actions (opening apps, typing, reading Calculator's display, web search, file ops), and talks back.

## Architecture

```
Mic → faster-whisper on CPU (local STT; VRAM stays free for the planner)
  → local two-tier router (qwen3.5:4b, on-device — no network call)
      ├─ chat    → answered directly by the fast tier, no planning loop
      ├─ simple  → fast tier plans (≤8 steps), escalates to 8B on failure
      └─ complex → qwen3:8b (local, Ollama) — structured-output planner
          → executes tools, loops until done
          → spoken reply via Kokoro-ONNX (local neural TTS, CPU)
```

Everything works fully offline by default. Groq is strictly opt-in (`ULTRON_TALK_BACKEND=groq`) — see [Privacy](#privacy) below for exactly what that changes.

**VRAM budget (4–8 GB):** STT, TTS, wake word, and embeddings all run on CPU.
Only the planner touches the GPU. `qwen3:8b` at Q4_K_M needs ~5 GB; on a
4 GB card set `ULTRON_MODEL=qwen3:4b` to stay in the 4B class. Ollama loads
the fast tier and the 8B tier on demand, so both can share the card.

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
- Verifies `qwen3:8b` (the default `ULTRON_MODEL`) and `qwen3.5:4b`
  (the default `ULTRON_FAST_MODEL`) are pulled — `ollama pull qwen3:8b`
  if you set it up manually
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

## Long-term memory

Ultron remembers across restarts. A local ChromaDB vector store (CPU-only ONNX MiniLM embeddings, ~90MB downloaded once on first run — no API key, no GPU) keeps two collections:

- **facts** — explicit long-lived facts ("remember my dog is Bruno"), saved via the `remember_fact` tool.
- **episodes** — every conversation turn, stored automatically.

Before planning, the agent semantically recalls relevant memories and injects them into the planner context. Tools: `remember_fact`, `recall_memory`, `forget_memory` — all auto-discovered like the rest.

```
pip install chromadb   # also added to requirements.txt
```

Env knobs:

```
$env:ULTRON_MEMORY_ENABLED="1"        # set "0" to disable
$env:ULTRON_MEMORY_DIR="$HOME/.ultron/memory"
$env:ULTRON_MEMORY_RECALL_K="5"
$env:ULTRON_MEMORY_MAX_DISTANCE="1.0" # cosine-distance cutoff for recall
```

If `chromadb` isn't installed, memory degrades gracefully — the agent runs exactly as before, minus recall.

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

## Configuration reference

| Variable | Default | What it does |
|---|---|---|
| `ULTRON_MODEL` | `qwen3:8b` | Tier-2 planner for complex multi-step tasks. On 4 GB VRAM use `qwen3:4b`. |
| `ULTRON_FAST_MODEL` | `qwen3.5:4b` | Tier-1: routing, chat replies, simple commands. |
| `ULTRON_ROUTER` | `local` | Two-tier routing. Set to `off` for single-model legacy behavior. |
| `ULTRON_FAST_MAX_STEPS` | `8` | Step budget for the fast tier before escalating to 8B. |
| `ULTRON_MAX_STEPS` | `30` | Step budget for the 8B planner. |
| `ULTRON_STT_MODEL` | `small` | faster-whisper model (CPU). `medium` is more accurate; disk is cheap. |
| `ULTRON_STT_DEVICE` | `cpu` | Set to `cuda` to move STT back to the GPU. |
| `ULTRON_HA_URL` | `http://homeassistant.local:8123` | Home Assistant base URL. |
| `ULTRON_HA_TOKEN` | (unset) | HA long-lived access token (Profile → Long-lived access tokens). |
| `ULTRON_NUDGE_WINDOW_MIN` | `30` | How far ahead reminder nudges look. |
| `ULTRON_TALK_BACKEND` | `local` | Set to `groq` for the opt-in cloud backend. |

## Proactive briefings

Ultron can brief you without being asked, via Windows Task Scheduler:

```powershell
.\setup_briefings.ps1
```

This registers two per-user tasks: a spoken **morning briefing** (8 AM daily —
date/time, weather, today's reminders) and **reminder nudges** (every
30 min, 8 AM–8 PM, silent when nothing is due). Manage reminders in
`~/.ultron/reminders.json`:

```json
[{"text": "Standup meeting", "time": "10:00"}]
```

The agent can also brief on demand: ask "what's my day look like?" and the
`get_morning_briefing` tool answers from the same source.

## Home Assistant

Free, local smart-home control — no cloud, no subscription:

```powershell
$env:ULTRON_HA_URL="http://homeassistant.local:8123"
$env:ULTRON_HA_TOKEN="your-long-lived-token"
```

Then "turn off the living room lights" just works: `ha_list_entities`
discovers names, `ha_get_state` reads them, `ha_call_service` controls them.

## Privacy

If `ULTRON_TALK_BACKEND=groq` is set, the raw text of every single utterance leaves your machine for classification, regardless of whether it ends up being handled locally or not. This is printed as a startup warning by the server itself, not just documented here. Set it to `local` (or leave it unset) to keep 100% of input on-device — nothing about tool execution, file access, or GUI control ever depends on Groq being enabled.

## Known limits worth knowing before you rely on this

- **No wall-clock timeout on the planner loop** — only a step count (`ULTRON_MAX_STEPS`). A confused model taking 30 slightly-different-but-still-wrong steps could take a long time before giving up.
- **GUI automation is a mix of deterministic UIAutomation** (e.g. `read_calculator_display`) where a dedicated tool exists, and vision/screenshot-based `capture_screen` as the general fallback elsewhere — the former is faster and more reliable; prefer building dedicated UIAutomation tools for any app you use often.
- **Tool control (file access, app launching, GUI input) is Windows-specific throughout** (`pywinauto`) — not portable to macOS/Linux without rewriting that layer.
