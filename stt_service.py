"""
Local STT service — Step 2 of the local AI agent build.

Tap the hotkey to START recording, tap it again to STOP.
On stop, audio is transcribed locally with faster-whisper and sent to
the agent server (Step 3) via send_to_agent().

Hotkey: Ctrl+Alt+Space  (change in HOTKEY below)
"""

import os
import sys

# --- Auto-fix: make the pip-installed cublas/cudnn DLLs discoverable ---
# Avoids having to manually prepend them to PATH every session.
_venv_root = sys.prefix
_nvidia_base = os.path.join(_venv_root, "Lib", "site-packages", "nvidia")
for _pkg in ("cublas", "cudnn"):
    _bin_path = os.path.join(_nvidia_base, _pkg, "bin")
    if os.path.isdir(_bin_path):
        os.environ["PATH"] = _bin_path + os.pathsep + os.environ.get("PATH", "")
# -------------------------------------------------------------------------

import queue
import tempfile
import threading
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# ---------------- Config ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
# Using small model for fast loading, but with language="en" forced for accuracy
MODEL_SIZE = "small"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
HOTKEY = {keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.space}

AGENT_ENDPOINT = "http://localhost:8000/speak"  # Step 3 agent server
AGENT_STATUS   = "http://localhost:8000/status"
# -----------------------------------------

print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE})...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded. Ready.")

recording = False
frames = []
audio_q = queue.Queue()
current_keys = set()


def audio_callback(indata, frames_count, time_info, status):
    if status:
        print(status)
    audio_q.put(indata.copy())


def record_loop():
    global recording, frames
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, dtype="float32"
    ):
        while True:
            chunk = audio_q.get()
            if recording:
                frames.append(chunk)


def save_wav(path, audio_frames):
    audio_data = np.concatenate(audio_frames, axis=0)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def transcribe(path):
    # language="en" forces English and reduces translation hallucinations.
    # condition_on_previous_text=False prevents it from inventing text out of background noise.
    segments, info = model.transcribe(path, beam_size=5, language="en", condition_on_previous_text=False)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def send_to_agent(text: str):
    print(f"[you said] {text}")
    if not AGENT_ENDPOINT:
        return
    import requests
    try:
        resp = requests.post(AGENT_ENDPOINT, json={"text": text}, timeout=60)
        data = resp.json()
        print(f"[agent] {data.get('reply', data)}")
    except Exception as e:
        print(f"[agent error] {e} (is agent_server.py running?)")


def is_agent_speaking() -> bool:
    """Returns True while the agent TTS is playing (used by wake word to avoid self-trigger)."""
    import requests as _req
    try:
        return _req.get(AGENT_STATUS, timeout=0.15).json().get("speaking", False)
    except Exception:
        return False


def toggle_recording():
    global recording, frames
    if not recording:
        print("\n🎙️  Recording... (press hotkey again to stop)")
        frames = []
        recording = True
    else:
        recording = False
        print("⏹️  Stopped. Transcribing...")
        if not frames:
            print("(no audio captured)")
            return
        captured_frames = list(frames)
        threading.Thread(target=_process_audio, args=(captured_frames,), daemon=True).start()

def _process_audio(captured_frames):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    save_wav(tmp_path, captured_frames)
    text = transcribe(tmp_path)
    os.remove(tmp_path)
    if text:
        send_to_agent(text)
    else:
        print("(nothing transcribed)")


hotkey_down = False  # tracks whether we've already fired for this physical press


def on_press(key):
    global hotkey_down
    current_keys.add(key)
    if HOTKEY.issubset(current_keys) and not hotkey_down:
        hotkey_down = True
        toggle_recording()


def on_release(key):
    global hotkey_down
    if key in current_keys:
        current_keys.remove(key)
    if not HOTKEY.issubset(current_keys):
        hotkey_down = False


if __name__ == "__main__":
    threading.Thread(target=record_loop, daemon=True).start()

    # Always-on wake word listener (say 'hey jarvis' to activate)
    # Runs alongside the hotkey — both call toggle_recording()
    from wake_word_service import start_wake_word_listener
    start_wake_word_listener(
        toggle_callback=toggle_recording,
        is_speaking_callback=is_agent_speaking,
    )

    # Start the hotkey listener in the background
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Local STT service running. Press Ctrl+Alt+Space or click the Ultron icon to start/stop recording.")
    print("Press Ctrl+C in terminal or close the Ultron window to quit.\n")

    # Start the UI blocking the main thread
    from ui_service import start_ui
    _wake_active = threading.Event()
    _wake_active.set()   # wake word is always active once started
    start_ui(
        toggle_callback=toggle_recording,
        get_recording_state_callback=lambda: recording,
        get_wake_active_callback=lambda: _wake_active.is_set() and not recording,
    )
