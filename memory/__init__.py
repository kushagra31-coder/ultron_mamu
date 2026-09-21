"""Long-term memory for Ultron.

Local-first vector memory backed by ChromaDB with CPU embeddings
(ONNX MiniLM — no API key, no GPU needed). Everything degrades gracefully:
if chromadb isn't installed or ULTRON_MEMORY_ENABLED=0, the store reports
itself unavailable and every method is a safe no-op.
"""
from __future__ import annotations

from .store import MemoryStore, get_store

__all__ = ["MemoryStore", "get_store"]
