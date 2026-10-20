"""User-tunable provider settings (``plugins.entries.web-deepseek.settings.*`` in config.yaml)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from hermes_deepseek_search.core import DEFAULT_BASE_URL, ValidationError
from hermes_deepseek_search.core.validation import validate_base_url

logger = logging.getLogger(__name__)

_MAX_CONCURRENCY = 25  # DeepSeek Fetch account QPS limit


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    min_rerank_score: float = 0.0  # drop search hits below this (0–3; 1.5+ = high quality per DeepSeek)
    max_retries: int = 2
    base_url: str = DEFAULT_BASE_URL
    extract_concurrency: int = 5

    @classmethod
    def from_ctx(cls, ctx: Any) -> ProviderSettings:
        """Read each key via ``ctx.get_config``; a missing/invalid value falls back to its default with a warning.
        Hermes builds without ``get_config`` (pre-settings releases) get all defaults."""
        get = getattr(ctx, "get_config", None)
        if not callable(get):
            return cls()
        defaults = cls()
        return cls(
            min_rerank_score=_coerce(get, "min_rerank_score", defaults.min_rerank_score, float, lo=0.0, hi=3.0),
            max_retries=_coerce(get, "max_retries", defaults.max_retries, int, lo=0, hi=5),
            base_url=_coerce_url(get, defaults.base_url),
            extract_concurrency=_coerce(
                get, "extract_concurrency", defaults.extract_concurrency, int, lo=1, hi=_MAX_CONCURRENCY
            ),
        )


def _coerce(get: Any, key: str, default: Any, kind: type, *, lo: float, hi: float) -> Any:
    raw = get(key, default)
    try:
        value = kind(raw)
    except (TypeError, ValueError, OverflowError):  # OverflowError: YAML `.inf` → int(float("inf"))
        logger.warning("web-deepseek setting %s=%r is not a valid %s; using %r", key, raw, kind.__name__, default)
        return default
    if isinstance(raw, bool) or (kind is int and isinstance(raw, float) and not raw.is_integer()):
        logger.warning("web-deepseek setting %s=%r is not a valid %s; using %r", key, raw, kind.__name__, default)
        return default
    if not lo <= value <= hi:  # also rejects NaN, which fails every comparison
        logger.warning("web-deepseek setting %s=%r is outside %s..%s; using %r", key, raw, lo, hi, default)
        return default
    return value


def _coerce_url(get: Any, default: str) -> str:
    """``base_url`` receives the API key: https only (http for localhost), no credentials in the URL. The raw
    value is never logged because it may embed a password."""
    raw = get("base_url", default)
    try:
        url = validate_base_url(raw if isinstance(raw, str) else "")
    except ValidationError as exc:
        logger.warning("web-deepseek setting base_url=<hidden> is invalid (%s); using %r", exc, default)
        return default
    host = urlsplit(url).hostname
    if host != urlsplit(default).hostname:
        logger.warning("web-deepseek setting base_url points at %s — the DeepSeek API key is sent there", host)
    return url
