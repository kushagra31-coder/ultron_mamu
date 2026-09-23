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
# Only needed when STT runs on CUDA. Kept for users who opt back into
# GPU transcription via ULTRON_STT_DEVICE=cuda.
_STT_DEVICE_PRE = os.getenv("ULTRON_STT_DEVICE", "cpu").strip().lower()
if _STT_DEVICE_PRE == "cuda":
    _venv_root = sys.prefix
    _nvidia_base = os.path.join(_venv_root, "Lib", "site-packages", "nvidia")
    for _pkg in ("cublas", "cudnn"):
        _bin_path = os.path.join(_nvidia_base, _pkg, "bin")
        if os.path.isdir(_bin_path):
            os.environ["PATH"] = _bin_path + os.pathsep + os.environ.get("PATH", "")
    del _venv_root, _nvidia_base, _bin_path, _pkg
del _STT_DEVICE_PRE
# -------------------------------------------------------------------------

import queue
import tempfile
import threading
import time
import uuid
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# ---------------- Config ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
# VRAM is reserved for the planner model, so STT runs on CPU. The "small"
# model with int8 compute is comfortably fast on a modern CPU; set
# ULTRON_STT_MODEL=medium for extra accuracy (disk is plentiful, CPU cost
# is a little higher). float16 does not run on CPU in CTranslate2 — int8
# is the recommended CPU compute type.
MODEL_SIZE = os.getenv("ULTRON_STT_MODEL", "small")
DEVICE = os.getenv("ULTRON_STT_DEVICE", "cpu").strip().lower()
COMPUTE_TYPE = os.getenv(
    "ULTRON_STT_COMPUTE_TYPE",
    "int8" if DEVICE == "cpu" else "float16",
)
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

# ---- Early-commit: stream partial transcripts while recording ------------
# Every PARTIAL_INTERVAL seconds the tail of the captured audio is
# transcribed (fast settings, no extra model/VRAM) and POSTed to the
# agent's /partial gate, which may execute a confident closed-set command
# BEFORE the user finishes speaking. Set ULTRON_EARLY_COMMIT=0 to disable.
EARLY_COMMIT     = os.getenv("ULTRON_EARLY_COMMIT", "1").strip() != "0"
PARTIAL_INTERVAL = float(os.getenv("ULTRON_PARTIAL_INTERVAL", "1.5"))
PARTIAL_WINDOW   = float(os.getenv("ULTRON_PARTIAL_WINDOW", "8"))
PARTIAL_ENDPOINT = "http://localhost:8000/partial"

_model_lock   = threading.Lock()  # faster-whisper: one transcription at a time
_session_id   = ""
_partial_seq  = 0
_partial_last = ""
_partial_stop = threading.Event()


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
    with _model_lock:
        segments, info = model.transcribe(path, beam_size=5, language="en", condition_on_previous_text=False)
        text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def _transcribe_audio_fast(audio):
    """Transcribe an in-memory audio chunk with fast settings (beam_size=1).
    Non-blocking: returns "" if the model is busy with the final transcription,
    so partials can never slow down or disturb the recording."""
    if not _model_lock.acquire(blocking=False):
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        segments, _info = model.transcribe(tmp_path, beam_size=1, language="en", condition_on_previous_text=False)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        os.remove(tmp_path)
        return text
    except Exception as exc:
        print(f"[partial] transcription failed: {exc}")
        return ""
    finally:
        _model_lock.release()


def _post_partial(text, session_id, seq):
    import requests as _req
    try:
        _req.post(PARTIAL_ENDPOINT,
                  json={"text": text, "session_id": session_id, "seq": seq},
                  timeout=0.6)
    except Exception:
        pass  # fire-and-forget: never disturb the recording


def _partial_loop():
    """While recording, transcribe the tail of the captured audio every
    PARTIAL_INTERVAL seconds and stream it to the agent's /partial gate."""
    global _partial_seq, _partial_last
    while recording and not _partial_stop.is_set():
        time.sleep(PARTIAL_INTERVAL)
        if not recording or _partial_stop.is_set():
            break
        snapshot = list(frames)
        if not snapshot:
            continue
        try:
            audio = np.concatenate(snapshot, axis=0).flatten()
        except Exception:
            continue
        max_samples = int(SAMPLE_RATE * PARTIAL_WINDOW)
        if len(audio) > max_samples:
            audio = audio[-max_samples:]
        if len(audio) < int(SAMPLE_RATE * 0.8):
            continue
        text = _transcribe_audio_fast(audio)
        if not text or text == _partial_last:
            continue
        _partial_last = text
        _partial_seq += 1
        threading.Thread(target=_post_partial,
                         args=(text, _session_id, _partial_seq),
                         daemon=True).start()


def send_to_agent(text: str, session_id: str = ""):
    print(f"[you said] {text}")
    if not AGENT_ENDPOINT:
        return
    import requests
    try:
        # Multi-step plans can legitimately take minutes; default the
        # client timeout high and allow override via ULTRON_AGENT_TIMEOUT.
        agent_timeout = float(os.getenv("ULTRON_AGENT_TIMEOUT", "300"))
        resp = requests.post(AGENT_ENDPOINT, json={"text": text, "session_id": session_id}, timeout=agent_timeout)
        data = resp.json()
        print(f"[agent] {data.get('reply', data)}")
    except Exception as e:
        print(f"[agent error] {e} (is agent_server.py running?)")


_speaking_cache = {"value": False, "at": 0.0}
_SPEAKING_CACHE_TTL = 0.5  # the wake-word loop asks ~12x/sec; cache it


def is_agent_speaking() -> bool:
    """Returns True while the agent TTS is playing (used by wake word to avoid self-trigger)."""
    import requests as _req
    now = time.monotonic()
    if now - _speaking_cache["at"] < _SPEAKING_CACHE_TTL:
        return _speaking_cache["value"]
    try:
        value = _req.get(AGENT_STATUS, timeout=0.15).json().get("speaking", False)
    except Exception:
        value = False
    _speaking_cache["value"] = value
    _speaking_cache["at"] = now
    return value


def toggle_recording():
    global recording, frames, _session_id, _partial_seq, _partial_last
    if not recording:
        print("\n🎙️  Recording... (press hotkey again to stop)")
        frames = []
        _session_id = uuid.uuid4().hex
        _partial_seq = 0
        _partial_last = ""
        _partial_stop.clear()
        recording = True
        if EARLY_COMMIT:
            threading.Thread(target=_partial_loop, daemon=True).start()
    else:
        recording = False
        _partial_stop.set()
        print("⏹️  Stopped. Transcribing...")
        if not frames:
            print("(no audio captured)")
            return
        captured_frames = list(frames)
        session_id = _session_id
        threading.Thread(target=_process_audio, args=(captured_frames, session_id), daemon=True).start()

def _process_audio(captured_frames, session_id=""):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    save_wav(tmp_path, captured_frames)
    text = transcribe(tmp_path)
    os.remove(tmp_path)
    if text:
        send_to_agent(text, session_id)
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
        is_recording_callback=lambda: recording,
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
