"""Make CLI smoke tests safe on a Windows cp1252 console.

The clinical text this project relays (FDA label sections, organism names) is
full of characters that the legacy Windows console codepage cannot encode -
Greek letters in pharmacology text, en-dashes, micro signs. Reconfiguring
stdout to UTF-8 with a replacement fallback keeps a smoke test from dying on a
print statement while leaving the underlying data untouched.
"""
from __future__ import annotations

import sys


def enable_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
