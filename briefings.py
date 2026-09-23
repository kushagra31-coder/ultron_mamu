"""Proactive briefings for Ultron — morning briefing + reminder nudges.

Designed to be triggered by Windows Task Scheduler (see setup_briefings.ps1),
but also usable by the agent itself via the `get_morning_briefing` tool.

Reminders live in ~/.ultron/reminders.json, a JSON list like:
    [
      {"text": "Standup meeting", "time": "10:00"},
      {"text": "Dentist", "date": "2026-09-25", "time": "17:30"}
    ]
"time" is HH:MM in the machine's local timezone. "date" is optional
(YYYY-MM-DD); entries without a date repeat daily.

Usage:
    python briefings.py morning [--speak]   # full morning briefing
    python briefings.py nudge   [--speak]   # reminders due in the next 30 min
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

REMINDERS_PATH = os.path.join(os.path.expanduser("~"), ".ultron", "reminders.json")
NUDGE_WINDOW_MIN = int(os.getenv("ULTRON_NUDGE_WINDOW_MIN", "30"))


def _ensure_reminders_file() -> str:
    os.makedirs(os.path.dirname(REMINDERS_PATH), exist_ok=True)
    if not os.path.exists(REMINDERS_PATH):
        with open(REMINDERS_PATH, "w", encoding="utf-8") as fh:
            json.dump([], fh, indent=2)
    return REMINDERS_PATH


def get_reminders() -> list[dict]:
    """Load reminders; return [] (and create the file) when absent/invalid."""
    _ensure_reminders_file()
    try:
        with open(REMINDERS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        return [r for r in data if isinstance(r, dict) and r.get("text")]
    except Exception:
        return []


def todays_reminders(now: _dt.datetime | None = None) -> list[dict]:
    """Reminders relevant today, sorted by time. Dated entries must match
    today; dateless entries repeat daily."""
    now = now or _dt.datetime.now()
    today = now.date().isoformat()
    out = []
    for r in get_reminders():
        date = (r.get("date") or "").strip()
        if date and date != today:
            continue
        out.append(r)
    out.sort(key=lambda r: (r.get("time") or "99:99"))
    return out


def due_soon(now: _dt.datetime | None = None) -> list[dict]:
    """Today's reminders whose time falls within the nudge window."""
    now = now or _dt.datetime.now()
    upcoming = []
    for r in todays_reminders(now):
        t = (r.get("time") or "").strip()
        try:
            hh, mm = (int(x) for x in t.split(":"))
            when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except ValueError:
            continue
        delta_min = (when - now).total_seconds() / 60
        if 0 <= delta_min <= NUDGE_WINDOW_MIN:
            upcoming.append((delta_min, r))
    upcoming.sort(key=lambda x: x[0])
    return [r for _, r in upcoming]


def get_weather() -> str | None:
    """Best-effort current weather via wttr.in (no API key). None on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://wttr.in/?format=%C,+%t,+wind+%w",
            headers={"User-Agent": "ultron-briefing"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8", "replace").strip()
    except Exception:
        return None


def _greeting(now: _dt.datetime) -> str:
    h = now.hour
    if h < 12:
        return "Good morning"
    if h < 17:
        return "Good afternoon"
    return "Good evening"


def build_morning_briefing(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now()
    parts = [
        f"{_greeting(now)}. It is {now.strftime('%A, %B %d, %I:%M %p')}.",
    ]
    weather = get_weather()
    parts.append(
        f"Current weather: {weather}." if weather
        else "I couldn't reach the weather service."
    )
    reminders = todays_reminders(now)
    if reminders:
        lines = "; ".join(
            f"{r['text']}" + (f" at {r['time']}" if r.get("time") else "")
            for r in reminders
        )
        parts.append(f"Today's reminders: {lines}.")
    else:
        parts.append("No reminders on the schedule today.")
    parts.append(f"Reminder file: {REMINDERS_PATH}")
    return " ".join(parts)


def build_nudge(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now()
    upcoming = due_soon(now)
    if not upcoming:
        return ""
    lines = "; ".join(
        f"{r['text']}" + (f" at {r['time']}" if r.get("time") else "")
        for r in upcoming
    )
    return f"Heads up — coming up in the next {NUDGE_WINDOW_MIN} minutes: {lines}."


def _speak(text: str) -> None:
    """Speak via Ultron's TTS when available; always print."""
    print(text)
    try:
        from tools.tts import speak, start_tts_worker
        start_tts_worker()
        speak(text)
    except Exception as exc:
        print(f"[briefing] TTS unavailable ({exc}); printed only.")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"morning", "nudge"}:
        print(__doc__.strip().splitlines()[-4:])
        return 2
    kind = argv[1]
    want_speak = "--speak" in argv
    text = build_morning_briefing() if kind == "morning" else build_nudge()
    if not text:
        print("(nothing to announce)")
        return 0
    if want_speak:
        _speak(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
