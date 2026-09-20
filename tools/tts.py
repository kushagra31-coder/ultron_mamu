"""Ultron-style Kokoro-ONNX voice.

Pipeline:
    kokoro-onnx -> sounddevice audio stream

The public interface stays compatible:
    speak_async(text)
    start_tts_worker()
    speak(text)
    is_speaking()
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading

# DirectML has a known ConvTranspose incompatibility with Kokoro ONNX, so ensure CPU execution
os.environ.setdefault("ONNX_PROVIDER", "CPUExecutionProvider")

import sounddevice as sd
from kokoro_onnx import Kokoro

QUEUE_LIMIT = 12
VOICE_NAME = os.getenv("ULTRON_TTS_VOICE", "am_adam")

_queue: queue.Queue[str] = queue.Queue(maxsize=QUEUE_LIMIT)
_started = False
_start_lock = threading.Lock()
_kokoro: Kokoro | None = None
_is_playing_audio = False


async def _render_and_play_async(text: str) -> None:
    global _is_playing_audio
    if _kokoro is None:
        print("[TTS] Error: Kokoro model not initialized.")
        return
    
    _is_playing_audio = True
    try:
        # Create a stream of audio chunks
        stream = _kokoro.create_stream(text, voice=VOICE_NAME, speed=1.0, lang="en-us")
        
        async for chunk, sample_rate in stream:
            # Play each chunk synchronously via sounddevice
            sd.play(chunk, samplerate=sample_rate)
            sd.wait()
            
    except Exception as e:
        print(f"[TTS] Error during playback: {e}")
    finally:
        _is_playing_audio = False


def _render_and_play(text: str) -> None:
    # Run the async stream consumer synchronously in this worker thread
    asyncio.run(_render_and_play_async(text))


def _worker() -> None:
    global _kokoro
    try:
        # Ensure paths exist relative to the project root
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "kokoro", "kokoro-v1.0.int8.onnx")
        voices_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "kokoro", "voices-v1.0.bin")
        
        _kokoro = Kokoro(model_path, voices_path)
        print(f"[TTS] Kokoro-ONNX voice selected: {VOICE_NAME}")
        print("[TTS] Ultron voice worker ready.")
    except Exception as e:
        print(f"[TTS] Failed to initialize Kokoro: {e}")
        return

    while True:
        text = _queue.get()
        try:
            if text:
                _render_and_play(text)
        except Exception as exc:
            print(f"[TTS error] {exc}")
        finally:
            _queue.task_done()


def start_tts_worker() -> None:
    global _started
    with _start_lock:
        if _started:
            return
        thread = threading.Thread(
            target=_worker,
            name="ultron-tts",
            daemon=True,
        )
        thread.start()
        _started = True


def speak_async(text: str) -> None:
    if not text:
        return

    start_tts_worker()

    try:
        _queue.put_nowait(text)
    except queue.Full:
        try:
            _queue.get_nowait()
            _queue.task_done()
        except queue.Empty:
            pass
        try:
            _queue.put_nowait(text)
        except queue.Full:
            print("[TTS] Queue full; dropping newest response.")


def speak(text: str) -> None:
    speak_async(text)


def is_speaking() -> bool:
    if not _started:
        return False
    # If the queue has items, we are processing speech
    if not _queue.empty():
        return True
    return _is_playing_audio
