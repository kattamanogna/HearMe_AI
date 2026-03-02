"""In-memory chat history storage per session."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from threading import Lock
from typing import Any

MAX_HISTORY_PER_SESSION = 10

_history_store: defaultdict[str, deque[dict[str, Any]]] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_PER_SESSION)
)
_store_lock = Lock()


def store_interaction(session_id: str, interaction: Mapping[str, Any]) -> None:
    """Store a single interaction for the given session.

    Keeps only the last ``MAX_HISTORY_PER_SESSION`` records per session.
    """

    normalized_session_id = session_id.strip() or "default"
    with _store_lock:
        _history_store[normalized_session_id].append(dict(interaction))


def get_chat_history(session_id: str) -> list[dict[str, Any]]:
    """Return chat history for a session as an ordered list."""

    normalized_session_id = session_id.strip() or "default"
    with _store_lock:
        return list(_history_store[normalized_session_id])
