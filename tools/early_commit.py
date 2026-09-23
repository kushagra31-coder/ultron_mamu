"""Early-commit gate: Jev-style preemptive execution for Ultron.

While the user is still speaking, the STT service POSTs partial transcripts
to /partial. This module decides — with zero model calls, so no extra VRAM
or latency — whether a partial is a confident, closed-set command that can
be executed RIGHT NOW instead of waiting for the finished utterance.

Rules (ported from the Jev voice-computer-use pattern):
- Act only on a small closed set of safe, non-destructive, reversible
  actions (opening allowlisted apps). Never on free text, dictation,
  file writes, or anything destructive.
- A pattern must match at the START of the partial (the user has finished
  saying the command part, even if the sentence continues).
- Each (session, action) commits at most once; stale/out-of-order partials
  are dropped via a per-session sequence number.
- Correction-as-undo: "no", "never mind", "undo", ... reverses the last
  early action instead of a speculative rollback (there is no speculation
  here — only confident commits).
"""
from __future__ import annotations

import os
import re
import threading
import time

from .registry import TOOLS

ENABLED = os.getenv("ULTRON_EARLY_COMMIT", "1").strip() != "0"

# Undo phrases: must be the WHOLE utterance, so "notepad" never matches "no".
_UNDO_RE = re.compile(
    r"^(no+|nope|never\s*mind|undo(\s+that)?|wrong(\s+one)?"
    r"|cancel(\s+that)?|stop(\s+that)?|don't(\s+do\s+that)?)[\s.,!]*$",
    re.IGNORECASE,
)

_UNDO_TTL = 90.0   # seconds an early action stays undoable
_UNDO_MAX = 10


def _open_app(app_key: str, focus_hint: str, label: str):
    """Build a closed-set entry for opening an allowlisted app."""
    pattern = re.compile(
        r"^(open|launch|start)\s+(the\s+)?"
        + re.escape(app_key).replace(r"\ ", r"\s+")
        + r"\b",
        re.IGNORECASE,
    )

    def args(_m: re.Match) -> dict:
        return {"application": app_key}

    def undo() -> list[tuple[str, dict]]:
        # Best effort: focus the window we opened, then close it.
        return [
            ("focus_window", {"title_contains": focus_hint}),
            ("hotkey", {"keys": "alt+f4"}),
        ]

    return (pattern, "open_application", args, undo, f"Opened {label}")


# Closed set: app launches only. Everything else waits for the full
# utterance and the normal planner pipeline.
_EARLY_ACTIONS = [
    _open_app("notepad", "Notepad", "Notepad"),
    _open_app("calc", "Calculator", "Calculator"),
    _open_app("calculator", "Calculator", "Calculator"),
    _open_app("code", "Visual Studio Code", "VS Code"),
    _open_app("vscode", "Visual Studio Code", "VS Code"),
    _open_app("vs code", "Visual Studio Code", "VS Code"),
    _open_app("visual studio code", "Visual Studio Code", "VS Code"),
    _open_app("brave", "Brave", "Brave"),
    _open_app("msedge", "Edge", "Edge"),
    _open_app("edge", "Edge", "Edge"),
    _open_app("microsoft edge", "Edge", "Edge"),
    _open_app("browser", "Brave", "browser"),
    _open_app("youtube", "YouTube", "YouTube"),
    _open_app("youtube music", "YouTube", "YouTube Music"),
    _open_app("gmail", "Gmail", "Gmail"),
    _open_app("github", "GitHub", "GitHub"),
    _open_app("chatgpt", "ChatGPT", "ChatGPT"),
    _open_app("leetcode", "LeetCode", "LeetCode"),
]

# open_file_explorer / open_windows_settings take no app key.
_EARLY_ACTIONS += [
    (
        re.compile(r"^(open|launch)\s+(the\s+)?(file\s+explorer|windows\s+explorer|explorer|this\s+pc)\b", re.IGNORECASE),
        "open_file_explorer",
        lambda _m: {},
        lambda: [("focus_window", {"title_contains": "File Explorer"}), ("hotkey", {"keys": "alt+f4"})],
        "Opened File Explorer",
    ),
    (
        re.compile(r"^(open|launch)\s+(the\s+)?(windows\s+)?settings\b", re.IGNORECASE),
        "open_windows_settings",
        lambda _m: {},
        lambda: [("focus_window", {"title_contains": "Settings"}), ("hotkey", {"keys": "alt+f4"})],
        "Opened Settings",
    ),
]


def _looks_ok(result: object) -> bool:
    """Same failure sniffing the planner uses — don't commit failed tools."""
    s = str(result or "").lower()
    return not (
        not s
        or "failed" in s
        or "fail-safe" in s
        or "could not" in s
        or "error" in s
        or "not allowed" in s
        or "windows-only" in s
        or s.startswith("tool '")
        or s.startswith("unknown tool")
    )


_lock = threading.Lock()
_sessions: dict[str, dict] = {}
_undo_stack: list[dict] = []


def _session(sid: str) -> dict:
    st = _sessions.get(sid)
    if st is None:
        st = {"seq": 0, "committed": set(), "notes": [], "started": time.monotonic()}
        _sessions[sid] = st
    # Forget sessions older than 10 minutes.
    if len(_sessions) > 40:
        cutoff = time.monotonic() - 600
        for k in [k for k, v in _sessions.items() if v["started"] < cutoff]:
            del _sessions[k]
    return st


def _run_tool(name: str, args: dict) -> str:
    spec = TOOLS.get(name)
    if spec is None:
        return f"Unknown tool: {name}"
    try:
        return str(spec.function(**args))
    except TypeError as exc:
        return f"Tool '{name}' received invalid arguments: {exc}"
    except Exception as exc:
        return f"Tool '{name}' failed: {exc}"


def do_undo() -> str | None:
    """Reverse the most recent early action. Returns a description or None."""
    now = time.monotonic()
    with _lock:
        while _undo_stack and now - _undo_stack[-1]["at"] > _UNDO_TTL:
            _undo_stack.pop()
        if not _undo_stack:
            return None
        entry = _undo_stack.pop()
    # Run the inverse outside the state lock.
    for tool_name, args in entry["undo_calls"]:
        if tool_name == "focus_window":
            res = _run_tool(tool_name, args)
            if not _looks_ok(res):
                print(f"[early] undo: could not focus '{args}' — skipping close")
                return None
        else:
            _run_tool(tool_name, args)
    print(f"[early] undone: {entry['desc']}")
    return entry["desc"]


def try_undo_full_text(text: str) -> str | None:
    """Check a FINAL transcript for an undo phrase. Returns undone desc or None."""
    if _UNDO_RE.match(text.strip()):
        return do_undo()
    return None


def handle_partial(text: str, session_id: str, seq: int) -> dict:
    """Evaluate one partial transcript. Returns {"committed": desc|None,
    "undone": desc|None}."""
    out = {"committed": None, "undone": None}
    if not ENABLED:
        return out
    text = (text or "").strip()
    if not text or not session_id:
        return out

    # Voice correction wins over everything.
    if _UNDO_RE.match(text):
        undone = do_undo()
        out["undone"] = undone
        return out

    with _lock:
        st = _session(session_id)
        if seq <= st["seq"]:
            return out  # stale / out-of-order partial
        st["seq"] = seq

        for pattern, tool_name, args_fn, undo_fn, desc in _EARLY_ACTIONS:
            m = pattern.match(text)
            if not m:
                continue
            action_key = f"{tool_name}:{sorted(args_fn(m).items())}"
            if action_key in st["committed"]:
                return out  # already acted on this command this session
            args = args_fn(m)
            st["committed"].add(action_key)
            break
        else:
            return out

    # Execute outside the state lock (tool calls can take a second).
    result = _run_tool(tool_name, args)
    if not _looks_ok(result):
        print(f"[early] '{desc}' failed: {result}")
        with _lock:
            st = _sessions.get(session_id)
            if st is not None:
                st["committed"].discard(action_key)
        return out

    print(f"[early] ⚡ committed early: {desc} (partial: {text!r})")
    with _lock:
        st = _sessions.get(session_id)
        if st is not None:
            st["notes"].append(desc)
        _undo_stack.append({"desc": desc, "undo_calls": undo_fn(), "at": time.monotonic()})
        del _undo_stack[: -_UNDO_MAX]
    out["committed"] = desc
    return out


def pop_session_note(session_id: str) -> str:
    """Return (and clear) the human-readable early-action note for /speak."""
    if not session_id:
        return ""
    with _lock:
        st = _sessions.pop(session_id, None)
    if not st or not st["notes"]:
        return ""
    return "; ".join(st["notes"])
