"""Search over the model's own organism vocabulary.

The trained model only knows the 245 organism names it saw during training
(`bundle.organisms`). This module filters that closed list; it never
invents, guesses, or falls back to a name the model was not trained on.
"""
from __future__ import annotations


def search_organisms(bundle, query: str) -> list[str]:
    """Case-insensitive substring/token match over the model's organisms."""
    q = (query or "").strip().upper()
    if not q:
        return list(bundle.organisms)
    tokens = q.split()
    return [o for o in bundle.organisms if all(t in o for t in tokens)]


def is_supported(bundle, name: str) -> bool:
    return name in set(bundle.organisms)
