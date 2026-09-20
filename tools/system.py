"""Windows application and path tools with a conservative allowlist."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from .registry import tool


# Explicitly allowlisted Windows applications and URLs.
# Values are executables passed to subprocess.Popen(), or URLs passed to os.startfile().
ALLOWED_APPS: dict[str, str] = {
    "code": "code",
    "notepad": "notepad",
    "calc": "calc",
    "brave": "brave",
    "msedge": "msedge",
    "explorer": "explorer",
    "youtube": "https://www.youtube.com",
    "youtube music": "https://music.youtube.com",
    "chatgpt": "https://chatgpt.com",
    "leetcode": "https://leetcode.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
}


# Common names produced by natural language or speech recognition.
# Each alias maps to an ALLOWED_APPS key.
_APP_ALIASES: dict[str, str] = {
    "calculator": "calc",
    "calc.exe": "calc",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "brave browser": "brave",
    "browser": "brave",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "file explorer": "explorer",
    "windows explorer": "explorer",
    "yt": "youtube",
    "yt music": "youtube music",
    "chat gpt": "chatgpt",
    "gpt": "chatgpt",
    "leet code": "leetcode",
}


def _normalize_application(application: str) -> str:
    """Normalize a requested application name to an allowlisted key."""
    key = application.strip().lower()
    return _APP_ALIASES.get(key, key)


def _safe_arguments(arguments: str) -> list[str] | str:
    """Parse optional Windows command-line arguments safely."""
    if not arguments.strip():
        return []

    try:
        return shlex.split(arguments, posix=False)
    except ValueError as exc:
        return f"Invalid application arguments: {exc}"


@tool(
    description=(
        "Launch one of the explicitly allowed Windows applications: "
        "code, notepad, calc, brave, msedge, explorer, or popular websites "
        "(youtube, youtube music, chatgpt, leetcode, github, gmail). Common names "
        "such as Calculator, VS Code, Edge, and File Explorer are accepted."
    )
)
def open_application(application: str, arguments: str = "") -> str:
    """Launch an explicitly allowlisted Windows application.

    Args:
        application: Application name such as calc, Calculator, notepad,
            VS Code, Brave, Edge, or File Explorer.
        arguments: Optional command-line arguments.
    """
    if os.name != "nt":
        return "This tool is Windows-only."

    original = application.strip()
    key = _normalize_application(original)

    if key not in ALLOWED_APPS:
        allowed = ", ".join(sorted(ALLOWED_APPS))
        return (
            f"'{original}' is not allowed. "
            f"Allowed applications: {allowed}."
        )

    parsed = _safe_arguments(arguments)
    if isinstance(parsed, str):
        return parsed

    executable = ALLOWED_APPS[key]

    if executable.startswith("http://") or executable.startswith("https://"):
        try:
            os.startfile(executable)
            return f"Opened {key} in the default browser."
        except OSError as exc:
            return f"Could not open '{key}': {exc}"

    try:
        subprocess.Popen(
            [executable, *parsed],
            close_fds=True,
        )
        return f"Launched {key}."
    except FileNotFoundError:
        return f"Windows could not find the '{key}' application/command."
    except OSError as exc:
        return f"Could not launch '{key}': {exc}"


@tool(description="Open a local file or folder using the Windows default application.")
def open_path(path: str) -> str:
    """Open a local path.

    Args:
        path: File or folder path. Relative paths are resolved from the user's
            home directory.
    """
    if os.name != "nt":
        return "This tool is Windows-only."

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.home() / p

    try:
        p = p.resolve()
    except OSError:
        return f"Could not resolve path: {p}"

    if not p.exists():
        return f"Path does not exist: {p}"

    try:
        os.startfile(str(p))
        return f"Opened {p}."
    except OSError as exc:
        return f"Could not open {p}: {exc}"


@tool(description="Return basic local Windows and Python runtime information.")
def system_info() -> str:
    """Return basic runtime information."""
    return (
        f"OS: {sys.platform}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Executable: {sys.executable}\n"
        f"Home: {Path.home()}"
    )
