"""Persistent long-term memory backed by ChromaDB (local, CPU embeddings).

Two collections:

- ``facts``: explicit long-lived facts/preferences, saved when the user says
  "remember ..." (or when the agent calls remember_fact on its own).
- ``episodes``: one entry per conversation turn (user text + agent reply),
  stored automatically by the /speak endpoint.

Configuration (env vars):

- ULTRON_MEMORY_ENABLED: "1" (default) or "0" to disable entirely.
- ULTRON_MEMORY_DIR: storage directory (default ~/.ultron/memory).
- ULTRON_MEMORY_RECALL_K: memories injected per turn (default 5).
- ULTRON_MEMORY_MAX_DISTANCE: cosine-distance cutoff for recall (default 1.0;
  0.0 = identical, 2.0 = opposite).

The default ChromaDB embedding function is ONNX MiniLM — CPU-only,
~90MB downloaded once into the HuggingFace cache on first run. No API key.

Everything degrades gracefully: missing chromadb, bad path, or any runtime
error disables the store (with a log line) instead of breaking the agent.
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime

ENABLED = os.getenv("ULTRON_MEMORY_ENABLED", "1") == "1"
MEMORY_DIR = os.getenv(
    "ULTRON_MEMORY_DIR", os.path.join(os.path.expanduser("~"), ".ultron", "memory")
)
RECALL_K = max(1, int(os.getenv("ULTRON_MEMORY_RECALL_K", "5")))
MAX_DISTANCE = float(os.getenv("ULTRON_MEMORY_MAX_DISTANCE", "1.0"))

_FACTS_COLLECTION = "ultron_facts"
_EPISODES_COLLECTION = "ultron_episodes"


def _utcnow() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MemoryStore:
    """Thread-safe wrapper around two ChromaDB collections."""

    def __init__(self, path: str = MEMORY_DIR):
        self._path = path
        self._lock = threading.Lock()
        self._client = None
        self._facts = None
        self._episodes = None
        self._available: bool | None = None  # None = not probed yet

    # -------------------------------------------------- lifecycle
    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_init()
        return self._available

    def _try_init(self) -> bool:
        if not ENABLED:
            return False
        try:
            import chromadb
        except ImportError:
            print(
                "[memory] chromadb not installed — long-term memory disabled. "
                "Install with: pip install chromadb"
            )
            return False
        try:
            os.makedirs(self._path, exist_ok=True)
            # Default embedding function = ONNX MiniLM on CPU, no API key.
            self._client = chromadb.PersistentClient(path=self._path)
            self._facts = self._client.get_or_create_collection(
                _FACTS_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
            self._episodes = self._client.get_or_create_collection(
                _EPISODES_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as exc:  # never break the agent over memory
            print(f"[memory] init failed — disabled: {exc}")
            return False

    # -------------------------------------------------- writes
    def add_fact(self, fact: str) -> str | None:
        """Store an explicit fact. Returns the memory id, or None."""
        fact = (fact or "").strip()
        if not fact or not self.available:
            return None
        with self._lock:
            try:
                mid = uuid.uuid4().hex
                self._facts.add(
                    ids=[mid],
                    documents=[fact],
                    metadatas=[{"ts": _utcnow(), "kind": "fact"}],
                )
                return mid
            except Exception as exc:
                print(f"[memory] add_fact failed: {exc}")
                return None

    def add_episode(self, user_text: str, agent_reply: str) -> str | None:
        """Store one conversation turn. Returns the memory id, or None."""
        user_text = (user_text or "").strip()
        agent_reply = (agent_reply or "").strip()
        if not user_text or not agent_reply or not self.available:
            return None
        with self._lock:
            try:
                mid = uuid.uuid4().hex
                doc = f"User: {user_text}\nUltron: {agent_reply}"
                self._episodes.add(
                    ids=[mid],
                    documents=[doc],
                    metadatas=[{"ts": _utcnow(), "kind": "episode"}],
                )
                return mid
            except Exception as exc:
                print(f"[memory] add_episode failed: {exc}")
                return None

    def forget(self, query: str) -> str | None:
        """Delete the single memory best matching query. Returns its text."""
        hits = self.recall(query, k=1)
        if not hits:
            return None
        hit = hits[0]
        with self._lock:
            try:
                if hit["kind"] == "fact":
                    self._facts.delete(ids=[hit["id"]])
                else:
                    self._episodes.delete(ids=[hit["id"]])
                return hit["text"]
            except Exception as exc:
                print(f"[memory] forget failed: {exc}")
                return None

    # -------------------------------------------------- reads
    def recall(self, query: str, k: int = RECALL_K) -> list[dict]:
        """Semantic search across facts + episodes. Best-first, distance-cut."""
        query = (query or "").strip()
        if not query or not self.available:
            return []
        with self._lock:
            try:
                merged: list[dict] = []
                for coll, kind in (
                    (self._facts, "fact"),
                    (self._episodes, "episode"),
                ):
                    res = coll.query(
                        query_texts=[query],
                        n_results=max(1, min(k, 10)),
                        include=["documents", "metadatas", "distances"],
                    )
                    docs = (res.get("documents") or [[]])[0]
                    metas = (res.get("metadatas") or [[]])[0]
                    dists = (res.get("distances") or [[]])[0]
                    ids = (res.get("ids") or [[]])[0]
                    for doc, meta, dist, mid in zip(docs, metas, dists, ids):
                        if float(dist) <= MAX_DISTANCE:
                            merged.append(
                                {
                                    "id": mid,
                                    "text": doc,
                                    "kind": kind,
                                    "distance": float(dist),
                                    "meta": meta or {},
                                }
                            )
                merged.sort(key=lambda r: r["distance"])
                return merged[:k]
            except Exception as exc:
                print(f"[memory] recall failed: {exc}")
                return []

    def recall_block(self, query: str, k: int = RECALL_K) -> str:
        """Formatted memory block for prompt injection ("" when none)."""
        hits = self.recall(query, k)
        if not hits:
            return ""
        lines = [
            "RELEVANT MEMORIES (use them to personalize your reply; "
            "never recite them unbidden):"
        ]
        for h in hits:
            ts = (h["meta"] or {}).get("ts", "")
            suffix = f" [{ts}]" if ts else ""
            lines.append(f"- [{h['kind']}] {h['text']}{suffix}")
        return "\n".join(lines)


# -------------------------------------------------- singleton
_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_store() -> MemoryStore:
    """Process-wide memory store (lazy, thread-safe)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MemoryStore()
    return _store


# -------------------------------------------------- self-test
if __name__ == "__main__":
    import sys
    import tempfile

    print("[memory] self-test: add -> recall -> forget")
    tmp = tempfile.mkdtemp(prefix="ultron-memory-test-")
    store = MemoryStore(path=tmp)
    if not store.available:
        print("[memory] self-test SKIPPED (chromadb unavailable)")
        sys.exit(2)
    fid = store.add_fact("The user's favourite editor is VS Code.")
    assert fid, "add_fact returned None"
    store.add_episode("remind me to call mom", "Done, I'll remind you.")
    hits = store.recall("what editor does the user like?")
    assert hits and "VS Code" in hits[0]["text"], f"recall missed: {hits}"
    print(f"[memory] recall hit: {hits[0]['text']!r} (d={hits[0]['distance']:.3f})")
    forgotten = store.forget("favourite editor")
    assert forgotten and "VS Code" in forgotten, "forget failed"
    assert not store.recall("favourite editor"), "memory not actually deleted"
    print("[memory] self-test PASSED")
