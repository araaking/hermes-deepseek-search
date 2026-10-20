"""User-tunable provider settings (``plugins.entries.web-deepseek.settings.*`` in config.yaml)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hermes_deepseek_search.core import DEFAULT_BASE_URL

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
    except (TypeError, ValueError):
        logger.warning("web-deepseek setting %s=%r is not a valid %s; using %r", key, raw, kind.__name__, default)
        return default
    if isinstance(raw, bool) or not lo <= value <= hi:
        logger.warning("web-deepseek setting %s=%r is outside %s..%s; using %r", key, raw, lo, hi, default)
        return default
    return value


def _coerce_url(get: Any, default: str) -> str:
    raw = get("base_url", default)
    if isinstance(raw, str) and raw.strip().lower().startswith(("https://", "http://")):
        return raw.strip()
    logger.warning("web-deepseek setting base_url=%r is not an http(s) URL; using %r", raw, default)
    return default
