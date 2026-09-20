"""Generic Windows keyboard/mouse/window-control tools for Ultron.

This module deliberately exposes reusable GUI primitives rather than
application-specific commands. The agent can compose these primitives after
launching an allowed application.
"""
from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes

from .registry import tool

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception as exc:  # optional tool module; loader can report the failure
    pyautogui = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""

_user32 = ctypes.windll.user32 if os.name == "nt" else None


def _check() -> str | None:
    """Return an environment error, or None when GUI control is available."""
    if os.name != "nt":
        return "This tool is Windows-only."
    if pyautogui is None:
        return f"PyAutoGUI is unavailable: {_IMPORT_ERROR}"
    return None


def _find_window(title_contains: str) -> int:
    """Find the first visible top-level window whose title contains text."""
    if _user32 is None:
        return 0

    match = title_contains.strip().lower()
    if not match:
        return 0

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )

    found = ctypes.c_void_p(0)

    def callback(hwnd: int, _lparam: int) -> bool:
        nonlocal found

        if not _user32.IsWindowVisible(hwnd):
            return True

        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.lower()

        if match in title:
            found = ctypes.c_void_p(hwnd)
            return False

        return True

    _user32.EnumWindows(EnumWindowsProc(callback), 0)
    return int(found.value or 0)


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Return (left, top, right, bottom) for a window."""
    if not hwnd or _user32 is None:
        return None

    rect = wintypes.RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None

    return (
        int(rect.left),
        int(rect.top),
        int(rect.right),
        int(rect.bottom),
    )


def _activate_window(hwnd: int) -> bool:
    """Restore and bring a window to the foreground."""
    if not hwnd or _user32 is None:
        return False

    # SW_RESTORE = 9
    _user32.ShowWindow(hwnd, 9)
    return bool(_user32.SetForegroundWindow(hwnd))


@tool(
    description=(
        "Wait for a Windows application/window to appear. Use this after "
        "open_application and before focusing or typing."
    )
)
def wait(seconds: float = 0.8) -> str:
    """Wait a short amount of time for a GUI to finish opening."""
    seconds = max(0.05, min(float(seconds), 10.0))
    time.sleep(seconds)
    return f"Waited {seconds:.2f} seconds."


@tool(
    description=(
        "Find and focus a visible Windows window whose title contains the "
        "given text. After focusing, move the mouse pointer to the center "
        "of the window so the interaction is visible on screen."
    )
)
def focus_window(title_contains: str) -> str:
    """Focus the first visible window matching part of its title."""
    error = _check()
    if error:
        return error

    hwnd = _find_window(title_contains)
    if not hwnd:
        return f"No visible window found containing '{title_contains}'."

    if not _activate_window(hwnd):
        return (
            f"Found '{title_contains}', but Windows did not allow "
            "focus to change."
        )

    rect = _get_window_rect(hwnd)
    if rect is not None:
        left, top, right, bottom = rect
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        try:
            pyautogui.moveTo(center_x, center_y, duration=0.10)
        except Exception as exc:
            return (
                f"Focused window matching '{title_contains}', but could not "
                f"move the mouse pointer: {exc}"
            )

        return (
            f"Focused window matching '{title_contains}' and moved the "
            f"mouse pointer to ({center_x}, {center_y})."
        )

    return f"Focused window matching '{title_contains}'."


@tool(
    description=(
        "Type ordinary text into the currently focused Windows application. "
        "Use after focusing the requested application. This sends real "
        "keyboard input through PyAutoGUI."
    )
)
def type_text(text: str, interval: float = 0.01) -> str:
    """Type text into the focused window."""
    error = _check()
    if error:
        return error

    if not text:
        return "No text was provided."

    interval = max(0.0, min(float(interval), 0.25))

    try:
        pyautogui.write(text, interval=interval)
    except Exception as exc:
        return f"Could not type text: {exc}"

    return f"Typed {len(text)} characters."


@tool(
    description=(
        "Press a keyboard key in the currently focused Windows application. "
        "Examples: enter, esc, tab, backspace, up, down, left, right."
    )
)
def press_key(key: str) -> str:
    """Press a single key."""
    error = _check()
    if error:
        return error

    key = key.strip().lower()

    allowed = {
        "enter",
        "esc",
        "escape",
        "tab",
        "backspace",
        "delete",
        "space",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "pageup",
        "pagedown",
        "insert",
        "win",
        "ctrl",
        "alt",
        "shift",
        *{f"f{i}" for i in range(1, 13)},
        *list("abcdefghijklmnopqrstuvwxyz0123456789"),
    }

    if key not in allowed:
        return f"Unsupported key '{key}'."

    actual_key = "escape" if key == "esc" else key

    try:
        pyautogui.press(actual_key)
    except Exception as exc:
        return f"Could not press {key}: {exc}"

    return f"Pressed {key}."


@tool(
    description=(
        "Press a '+' separated keyboard hotkey such as ctrl+l, ctrl+c, "
        "ctrl+v, alt+tab, or win+r in the currently focused application."
    )
)
def hotkey(keys: str) -> str:
    """Press a '+' separated keyboard combination."""
    error = _check()
    if error:
        return error

    parts = [part.strip().lower() for part in keys.split("+") if part.strip()]

    if not parts:
        return "No hotkey was provided."

    if len(parts) > 5:
        return "Hotkeys are limited to five keys."

    try:
        pyautogui.hotkey(*parts)
    except Exception as exc:
        return f"Could not press hotkey '{keys}': {exc}"

    return f"Pressed {keys}."


@tool(
    description=(
        "Move the Windows mouse pointer to screen coordinates x,y. "
        "Use when a visible cursor movement is required before clicking."
    )
)
def move_mouse(x: int, y: int) -> str:
    """Move the mouse pointer to screen coordinates."""
    error = _check()
    if error:
        return error

    try:
        x = int(x)
        y = int(y)
        pyautogui.moveTo(x, y, duration=0.10)
    except Exception as exc:
        return f"Could not move mouse: {exc}"

    return f"Moved mouse to ({x}, {y})."


@tool(
    description=(
        "Click the Windows desktop at screen coordinates x,y. "
        "Use only when the requested UI target is known to be at that "
        "location. The pointer is moved to the target before clicking."
    )
)
def click(
    x: int,
    y: int,
    button: str = "left",
    clicks: int = 1,
) -> str:
    """Move to and click a screen location."""
    error = _check()
    if error:
        return error

    button = button.lower().strip()
    if button not in {"left", "right", "middle"}:
        return "Button must be left, right, or middle."

    clicks = max(1, min(int(clicks), 3))
    x = int(x)
    y = int(y)

    try:
        pyautogui.moveTo(x, y, duration=0.10)
        pyautogui.click(
            x=x,
            y=y,
            clicks=clicks,
            button=button,
        )
    except Exception as exc:
        return f"Could not click at ({x}, {y}): {exc}"

    return f"Clicked at ({x}, {y})."


@tool(
    description=(
        "Scroll the active Windows application vertically. Positive values "
        "scroll up; negative values scroll down."
    )
)
def scroll(amount: int) -> str:
    """Scroll the active window."""
    error = _check()
    if error:
        return error

    amount = max(-20, min(int(amount), 20))

    try:
        pyautogui.scroll(amount)
    except Exception as exc:
        return f"Could not scroll: {exc}"

    return f"Scrolled {amount}."


@tool(
    description=(
        "Read the current numeric result displayed on the Windows Calculator. "
        "Use this instead of capture_screen when you need to read the Calculator result."
    )
)
def read_calculator_display() -> str:
    """Read the result from the Windows Calculator."""
    try:
        from pywinauto.application import Application
    except ImportError:
        return "pywinauto is not installed. Please install it to read the Calculator display."
        
    try:
        # Connect to the running Calculator
        app = Application(backend="uia").connect(title_re=".*Calculator.*", timeout=3)
        dlg = app.window(title_re=".*Calculator.*")
        
        # Find the result element (Windows 10/11 Calculator)
        results = dlg.child_window(auto_id="CalculatorResults")
        text = results.window_text()
        
        # Text usually looks like "Display is 52"
        if text and text.startswith("Display is "):
            text = text.replace("Display is ", "").strip()
            
        return f"Calculator display shows: {text}"
    except Exception as exc:
        return f"Could not read Calculator display: {exc}"

