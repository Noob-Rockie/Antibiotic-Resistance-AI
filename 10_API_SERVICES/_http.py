"""Shared HTTP plumbing for the external service clients.

Design rules enforced here (see docs/API_INTEGRATION.md):
  * Every call has a timeout and never raises into the caller.
  * Every response is wrapped in an `ApiResult` that records the source URL and
    the UTC retrieval timestamp, so the UI can always show provenance.
  * Responses are cached on disk so repeated lookups do not hammer public APIs.
  * A single environment switch (`ENABLE_LIVE_API=0`) disables all network I/O;
    clients then return `ApiResult(ok=False, reason="live API disabled")`.
  * No API key is ever hard-coded. Keys come from the environment only.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CACHE_DIR = Path(__file__).resolve().parents[1] / "01_DATA" / "interim" / "api_cache"
DEFAULT_TIMEOUT = 12
USER_AGENT = "Antibiotic-Resistance-AI/1.0 (academic research prototype)"


def live_enabled() -> bool:
    return os.getenv("ENABLE_LIVE_API", "1").strip() not in {"0", "false", "False", ""}


def cache_ttl_seconds() -> int:
    try:
        return int(float(os.getenv("API_CACHE_TTL_HOURS", "24")) * 3600)
    except ValueError:
        return 24 * 3600


@dataclass
class ApiResult:
    """Uniform envelope for every external lookup.

    `ok=False` is a normal, expected state - the UI renders a
    "not available - verify locally" message rather than inventing a value.
    """
    ok: bool
    data: Any = None
    source: str = ""
    source_url: str = ""
    retrieved_at_utc: str = ""
    from_cache: bool = False
    reason: str = ""
    disclaimer: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _cache_path(url: str, params: dict | None) -> Path:
    blob = url + "|" + json.dumps(params or {}, sort_keys=True)
    return CACHE_DIR / (hashlib.sha256(blob.encode()).hexdigest()[:32] + ".json")


def get_json(url: str, params: dict | None = None, *, source: str,
             timeout: int = DEFAULT_TIMEOUT,
             use_cache: bool = True) -> ApiResult:
    """GET a JSON endpoint, with disk caching and total failure containment."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cpath = _cache_path(url, params)

    if use_cache and cpath.exists():
        age = time.time() - cpath.stat().st_mtime
        if age < cache_ttl_seconds():
            try:
                payload = json.loads(cpath.read_text(encoding="utf-8"))
                return ApiResult(ok=True, data=payload["data"], source=source,
                                 source_url=payload.get("url", url),
                                 retrieved_at_utc=payload.get("retrieved_at_utc", ""),
                                 from_cache=True)
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt cache entry - fall through to a live fetch

    if not live_enabled():
        return ApiResult(ok=False, source=source, source_url=url,
                         retrieved_at_utc=now,
                         reason="Live API access is disabled (ENABLE_LIVE_API=0).")

    try:
        resp = requests.get(url, params=params, timeout=timeout,
                            headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        return ApiResult(ok=False, source=source, source_url=url,
                         retrieved_at_utc=now,
                         reason=f"Network error contacting {source}: "
                                f"{type(exc).__name__}")

    if resp.status_code == 404:
        return ApiResult(ok=False, source=source, source_url=resp.url,
                         retrieved_at_utc=now,
                         reason=f"{source} returned no record for this query.")
    if resp.status_code == 429:
        return ApiResult(ok=False, source=source, source_url=resp.url,
                         retrieved_at_utc=now,
                         reason=f"{source} rate limit reached. Try again later.")
    if not resp.ok:
        return ApiResult(ok=False, source=source, source_url=resp.url,
                         retrieved_at_utc=now,
                         reason=f"{source} returned HTTP {resp.status_code}.")

    try:
        data = resp.json()
    except ValueError:
        return ApiResult(ok=False, source=source, source_url=resp.url,
                         retrieved_at_utc=now,
                         reason=f"{source} returned a non-JSON response.")

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(
                {"data": data, "url": resp.url, "retrieved_at_utc": now}),
                encoding="utf-8")
        except OSError:
            pass  # caching is best-effort; never fail a lookup because of it

    return ApiResult(ok=True, data=data, source=source, source_url=resp.url,
                     retrieved_at_utc=now, from_cache=False)
