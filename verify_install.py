"""Windows-first preflight verification for Ultron."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if detail:
        print(f"       {detail}")
    return bool(ok)


def main() -> int:
    results: list[bool] = []

    results.append(check("Windows", os.name == "nt", f"os.name={os.name}"))

    exact_311 = sys.version_info[:2] == (3, 11)
    results.append(check("Python 3.11", exact_311, platform.python_version()))
    results.append(check("64-bit Python", sys.maxsize > 2**32, f"pointer_bits={8 * __import__('struct').calcsize('P')}") )

    modules = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "ollama",
        "ddgs",
        "pyttsx3",
        "faster_whisper",
        "sounddevice",
        "pynput",
        "numpy",
    ]

    for module in modules:
        try:
            importlib.import_module(module)
            results.append(check(f"Import {module}", True))
        except Exception as exc:
            results.append(check(f"Import {module}", False, f"{type(exc).__name__}: {exc}"))

    ollama_exe = shutil.which("ollama")
    results.append(check("Ollama on PATH", bool(ollama_exe), ollama_exe or "ollama.exe not found"))

    model = os.getenv("ULTRON_MODEL", "qwen2.5-coder:7b")
    if ollama_exe:
        try:
            proc = subprocess.run(
                [ollama_exe, "list"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            ok = any(
                line.split()[0] == model
                for line in proc.stdout.splitlines()
                if line.strip()
                and not line.upper().startswith("NAME")
                and len(line.split()) >= 1
            )
            results.append(
                check(
                    f"Model {model}",
                    ok,
                    "Run: ollama pull " + model if not ok else "installed",
                )
            )
        except Exception as exc:
            results.append(check("Ollama model check", False, str(exc)))

    try:
        # Ensure this project itself is being imported, not another similarly
        # named package elsewhere on sys.path.
        sys.path.insert(0, str(PROJECT))
        import tools.loader as loader
        import tools.registry as registry

        loader.load_tools()

        for name, err in sorted(loader.LOAD_ERRORS.items()):
            print(f"[FAIL] Tool module {name}")
            print(f"       {err}")
            results.append(False)

        count = len(registry.TOOLS)
        results.append(
            check(
                "Dynamic tool discovery",
                count >= 1 and not loader.LOAD_ERRORS,
                f"{count} tools loaded",
            )
        )
    except Exception as exc:
        results.append(check("Dynamic tool discovery", False, f"{type(exc).__name__}: {exc}"))

    print(f"\nResult: {sum(results)}/{len(results)} checks passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
