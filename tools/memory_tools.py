"""Long-term memory tools — local ChromaDB vector store, CPU embeddings.

Auto-discovered by tools/loader.py like every other tool module. If chromadb
isn't installed (or memory is disabled), the tools stay registered but report
that memory is unavailable instead of failing.
"""
from __future__ import annotations

from .registry import tool
from memory import get_store


@tool(description="Save a long-lived fact or preference about the user. Use whenever the user says 'remember ...'.")
def remember_fact(fact: str) -> str:
    """Save a fact to long-term memory.

    Args:
        fact: The fact to remember, e.g. 'User prefers dark mode' or "User's dog is named Bruno".
    """
    store = get_store()
    if not store.available:
        return "Long-term memory is unavailable (chromadb not installed or ULTRON_MEMORY_ENABLED=0)."
    mid = store.add_fact(fact)
    return f"Remembered: {fact}" if mid else "I couldn't save that fact."


@tool(description="Search long-term memory for facts or past conversations relevant to a query.")
def recall_memory(query: str, limit: int = 5) -> str:
    """Search long-term memory.

    Args:
        query: What to look for, e.g. 'user food preferences'.
        limit: Maximum memories to return (default 5).
    """
    store = get_store()
    if not store.available:
        return "Long-term memory is unavailable (chromadb not installed or ULTRON_MEMORY_ENABLED=0)."
    hits = store.recall(query, k=max(1, min(int(limit or 5), 10)))
    if not hits:
        return "No relevant memories found."
    return "\n".join(f"- [{h['kind']}] {h['text']}" for h in hits)


@tool(description="Delete the single memory that best matches a query.")
def forget_memory(query: str) -> str:
    """Forget one memory.

    Args:
        query: Description of the memory to delete, e.g. 'user favourite editor'.
    """
    store = get_store()
    if not store.available:
        return "Long-term memory is unavailable (chromadb not installed or ULTRON_MEMORY_ENABLED=0)."
    forgotten = store.forget(query)
    return f"Forgot: {forgotten}" if forgotten else "No matching memory found."
