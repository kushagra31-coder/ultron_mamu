"""Briefing tool — lets the agent answer 'what's my day look like?' itself."""
from __future__ import annotations

import importlib.util
import os

from .registry import tool


def _load_briefings():
    # Import the top-level briefings.py without polluting sys.modules on failure.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "briefings.py")
    spec = importlib.util.spec_from_file_location("ultron_briefings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@tool(
    description=(
        "Get Ultron's morning briefing: current date/time, weather, and "
        "today's reminders from the local reminders file. Use when the user "
        "asks what's on their schedule or what their day looks like."
    )
)
def get_morning_briefing() -> str:
    """Build the spoken morning briefing."""
    try:
        briefings = _load_briefings()
    except Exception as exc:
        return f"Could not load the briefing module: {exc}"
    try:
        return briefings.build_morning_briefing()
    except Exception as exc:
        return f"Could not build the briefing: {exc}"
