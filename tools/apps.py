"""App-specific Windows automation tools for Ultron.

Where tools/computer.py exposes generic GUI primitives (click, type, focus),
this module offers one deterministic tool per daily app, driven by
pywinauto's UIA backend. Deterministic beats screenshot-guessing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

from .registry import tool


def _windows_only() -> str | None:
    if os.name != "nt":
        return "This tool is Windows-only."
    return None


def _pywinauto():
    try:
        from pywinauto.application import Application
        return Application
    except ImportError:
        return None


@tool(
    description=(
        "Open Windows Notepad and type the given text into it. If save_path "
        "is provided, saves the file there (e.g. "
        "'C:\\\\Users\\\\akaza\\\\notes.txt'); otherwise leaves it unsaved."
    )
)
def notepad_write(text: str, save_path: str = "") -> str:
    """Open Notepad, type text, optionally save to a file."""
    error = _windows_only()
    if error:
        return error
    Application = _pywinauto()
    if Application is None:
        return "pywinauto is not installed."

    try:
        app = Application(backend="uia").start("notepad.exe")
        dlg = app.window(title_re=".*Notepad.*")
        dlg.wait("ready", timeout=10)
        edit = dlg.child_window(control_type="Edit")
        # type_keys is faster and more reliable than char-by-char for long text
        edit.type_keys(text, with_spaces=True)
        if save_path:
            dlg.type_keys("^s")
            time.sleep(0.5)
            save_dlg = app.window(title_re=".*Save.*")
            save_dlg.wait("ready", timeout=5)
            name_edit = save_dlg.child_window(control_type="Edit")
            name_edit.set_text(save_path)
            save_dlg.type_keys("{ENTER}")
            time.sleep(0.5)
            return f"Wrote {len(text)} characters to Notepad and saved to {save_path}."
        return f"Wrote {len(text)} characters to Notepad (unsaved)."
    except Exception as exc:
        return f"Could not write in Notepad: {exc}"


@tool(
    description=(
        "Open Windows File Explorer at the given folder path "
        "(e.g. 'C:\\\\Users\\\\akaza\\\\Downloads'). Opens This PC when "
        "no path is given."
    )
)
def open_file_explorer(path: str = "") -> str:
    """Open File Explorer at a folder."""
    error = _windows_only()
    if error:
        return error
    target = path.strip() or "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}"  # This PC
    try:
        os.startfile(target)  # noqa: S606 — user-directed local path
        return f"Opened File Explorer at '{path or 'This PC'}'."
    except Exception as exc:
        return f"Could not open File Explorer: {exc}"


@tool(
    description=(
        "Open a URL in the user's default web browser "
        "(e.g. 'https://www.youtube.com')."
    )
)
def open_url_in_browser(url: str) -> str:
    """Open a URL in the default browser."""
    error = _windows_only()
    if error:
        return error
    url = url.strip()
    if not url:
        return "No URL was provided."
    if "://" not in url:
        url = "https://" + url
    try:
        os.startfile(url)  # noqa: S606 — user-directed URL
        return f"Opened {url} in the default browser."
    except Exception as exc:
        return f"Could not open URL: {exc}"


@tool(
    description=(
        "Open a file or folder in Visual Studio Code "
        "(e.g. 'C:\\\\Users\\\\akaza\\\\code\\\\ultron_mamu'). "
        "Requires the 'code' command on PATH."
    )
)
def vscode_open(path: str) -> str:
    """Open a path in VS Code."""
    error = _windows_only()
    if error:
        return error
    path = path.strip()
    if not path:
        return "No path was provided."
    if shutil.which("code") is None:
        return "VS Code's 'code' command was not found on PATH."
    try:
        subprocess.Popen(["code", path])
        return f"Opened '{path}' in VS Code."
    except Exception as exc:
        return f"Could not open VS Code: {exc}"


@tool(
    description=(
        "Open a Windows Settings page, e.g. 'bluetooth', 'wifi', 'display', "
        "'sound', 'apps', 'windowsupdate'. Opens the Settings home page "
        "when no page is given."
    )
)
def open_windows_settings(page: str = "") -> str:
    """Open a Windows Settings page via its ms-settings: URI."""
    error = _windows_only()
    if error:
        return error
    page = page.strip().lower().replace(" ", "")
    uri = f"ms-settings:{page}" if page else "ms-settings:"
    try:
        os.startfile(uri)  # noqa: S606 — well-known Settings URI scheme
        return f"Opened Windows Settings ({page or 'home'})."
    except Exception as exc:
        return f"Could not open Settings: {exc}"


@tool(
    description="Lock the Windows workstation (Win+L)."
)
def lock_workstation() -> str:
    """Lock the screen."""
    error = _windows_only()
    if error:
        return error
    try:
        import ctypes
        ctypes.windll.user32.LockWorkStation()
        return "Workstation locked."
    except Exception as exc:
        return f"Could not lock the workstation: {exc}"
