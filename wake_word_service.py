"""
Always-on wake word listener for Ultron.

Uses openWakeWord with hey_jarvis (pre-trained, works today) as the
default activation phrase.  To use a custom "wake up ultron" model once
trained, set:  ULTRON_WAKE_MODEL=path/to/wake_up_ultron.onnx

Design decisions
----------------
- Detection is PAUSED while TTS is playing to prevent Ultron hearing itself.
- Sensitivity is tunable via ULTRON_WAKE_SENSITIVITY (0.0-1.0, default 0.5).
- If openwakeword is not installed, logs a warning and exits gracefully.
- The hotkey in stt_service.py remains fully functional as a manual override.

Wake phrase: say "hey jarvis" to activate.
To train a custom "wake up ultron" model (no voice recording needed):
  https://github.com/dscripka/openWakeWord#training-new-models
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd

WAKE_MODEL    = os.getenv("ULTRON_WAKE_MODEL", "hey_jarvis")
SENSITIVITY   = float(os.getenv("ULTRON_WAKE_SENSITIVITY", "0.5"))
SAMPLE_RATE   = 16000
CHUNK_SAMPLES = 1280   # 80 ms — openWakeWord's required chunk size

# ---- suppress noisy import logs ----
import logging
logging.getLogger("openwakeword").setLevel(logging.ERROR)


def _resolve_model_path(name_or_path: str) -> str | None:
    """Accept either a bare model name ('hey_jarvis') or a full .onnx path."""
    if os.path.isfile(name_or_path):
        return name_or_path
    # Try the bundled resources directory
    try:
        import openwakeword
        import pathlib
        model_dir = pathlib.Path(openwakeword.__file__).parent / "resources" / "models"
        # Match name with or without version suffix and extension
        for candidate in model_dir.glob("*.onnx"):
            stem = candidate.stem.lower()           # e.g. "hey_jarvis_v0.1"
            base = stem.split("_v")[0]              # e.g. "hey_jarvis"
            if base == name_or_path.lower() or stem == name_or_path.lower():
                return str(candidate)
    except Exception:
        pass
    return None


def start_wake_word_listener(
    toggle_callback: Callable[[], None],
    is_speaking_callback: Callable[[], bool],
) -> None:
    """
    Start the always-on wake word listener in a background daemon thread.

    Args:
        toggle_callback:    Called when wake word detected — same function
                            the hotkey calls to start recording.
        is_speaking_callback: Returns True while TTS is playing; detection
                            is paused during this window.
    """
    try:
        from openwakeword.model import Model as _OWWModel
    except ImportError:
        print("[wake] openwakeword not installed — wake word disabled.")
        print("[wake] Install with: pip install openwakeword")
        return

    model_path = _resolve_model_path(WAKE_MODEL)
    if model_path is None:
        print(f"[wake] Model '{WAKE_MODEL}' not found — wake word disabled.")
        print("[wake] Run: python -c \"from openwakeword.utils import download_models; download_models()\"")
        return

    def _listener():
        try:
            oww = _OWWModel(wakeword_models=[model_path], inference_framework="onnx")
        except Exception as exc:
            print(f"[wake] Failed to load model: {exc}")
            return

        model_key = list(oww.models.keys())[0]
        print(f"[wake] Listening for wake word '{WAKE_MODEL}' (sensitivity={SENSITIVITY})")
        print("[wake] Say 'hey jarvis' to activate Ultron.")

        audio_q: queue.Queue[np.ndarray] = queue.Queue()

        def _audio_cb(indata, frames, time_info, status):
            audio_q.put(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SAMPLES,
            dtype="float32",
            callback=_audio_cb,
        ):
            while True:
                chunk = audio_q.get()

                # Pause detection while TTS is speaking
                if is_speaking_callback():
                    continue

                # openWakeWord expects int16 PCM
                chunk_int16 = (chunk * 32767).astype(np.int16)
                prediction = oww.predict(chunk_int16)
                score = prediction.get(model_key, 0.0)

                if score >= SENSITIVITY:
                    print(f"[wake] Detected! score={score:.3f} — activating Ultron")
                    oww.reset()   # clear state so it won't re-trigger immediately
                    toggle_callback()
                    # Brief cooldown so it doesn't double-fire
                    time.sleep(1.5)

    t = threading.Thread(target=_listener, name="ultron-wake", daemon=True)
    t.start()


# ---- Standalone test -------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("Wake word standalone test")
    print("Say 'hey jarvis' — you have 30 seconds.")

    detected = threading.Event()

    def _on_detect():
        print("[TEST] Wake word detected! Test passed.")
        detected.set()

    start_wake_word_listener(
        toggle_callback=_on_detect,
        is_speaking_callback=lambda: False,
    )

    if not detected.wait(timeout=30):
        print("[TEST] No detection in 30s — check your mic and sensitivity.")
        sys.exit(1)
