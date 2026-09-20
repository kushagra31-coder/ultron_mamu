"""User-home filesystem tools."""
from __future__ import annotations
from pathlib import Path
from .registry import tool
HOME = Path.home().resolve()

def _resolve(path: str) -> Path:
    p=Path(path).expanduser()
    if not p.is_absolute(): p=HOME/p
    return p.resolve()

def _allowed(p: Path) -> bool:
    try: p.relative_to(HOME); return True
    except ValueError: return False

@tool(description="List files and folders under the user's home directory.")
def list_directory(path: str = ".") -> str:
    """List a directory.

    Args:
        path: Directory path.
    """
    p=_resolve(path)
    if not _allowed(p): return "Access denied: path must be inside the user home directory."
    if not p.is_dir(): return f"Not a directory: {p}"
    items=sorted(p.iterdir(), key=lambda x:(not x.is_dir(), x.name.lower()))
    return "\n".join(("[DIR] " if x.is_dir() else "[FILE] ")+x.name for x in items[:200]) or f"{p} is empty."

@tool(description="Search for files by name fragment under the user's home directory.")
def find_files(name_contains: str, root: str = ".") -> str:
    """Find files by name fragment.

    Args:
        name_contains: Case-insensitive filename fragment.
        root: Search root.
    """
    if not name_contains.strip(): return "A filename fragment is required."
    base=_resolve(root)
    if not _allowed(base) or not base.is_dir(): return "Invalid or disallowed search root."
    needle=name_contains.casefold(); out=[]
    for p in base.rglob("*"):
        if p.is_file() and needle in p.name.casefold():
            out.append(str(p))
            if len(out)>=100: break
    return "\n".join(out) if out else f"No files matched '{name_contains}'."

@tool(description="Read a UTF-8 text file under the user's home directory.")
def read_text_file(file_path: str, max_chars: int = 20000) -> str:
    """Read a text file.

    Args:
        file_path: File path.
        max_chars: Maximum characters returned.
    """
    p=_resolve(file_path)
    if not _allowed(p): return "Access denied: file must be inside the user home directory."
    if not p.is_file(): return f"File does not exist: {p}"
    try: text=p.read_text(encoding="utf-8")
    except UnicodeDecodeError: return "The file is not valid UTF-8 text."
    except OSError as exc: return f"Could not read file: {exc}"
    n=max(100,min(int(max_chars),100000))
    return text if len(text)<=n else text[:n]+f"\n\n[truncated at {n} characters]"

@tool(description="Write or overwrite a UTF-8 text file under the user's home directory.")
def write_text_file(file_path: str, content: str, overwrite: bool = True) -> str:
    """Write a text file.

    Args:
        file_path: File path.
        content: Complete file contents.
        overwrite: Whether an existing file may be replaced.
    """
    p=_resolve(file_path)
    if not _allowed(p): return "Access denied: file must be inside the user home directory."
    if p.exists() and not overwrite: return f"File already exists and overwrite=false: {p}"
    try: p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content,encoding="utf-8")
    except OSError as exc: return f"Could not write file: {exc}"
    return f"Wrote {len(content)} characters to {p}."

@tool(description="Create a folder under the user's home directory.")
def create_folder(path: str) -> str:
    """Create a folder.

    Args:
        path: Directory path.
    """
    p=_resolve(path)
    if not _allowed(p): return "Access denied: folder must be inside the user home directory."
    try: p.mkdir(parents=True, exist_ok=True)
    except OSError as exc: return f"Could not create folder: {exc}"
    return f"Folder ready: {p}"
