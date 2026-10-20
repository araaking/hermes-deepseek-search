"""Async client for the DeepSeek Search & Fetch API (``/v1/tools/search``, ``/v1/tools/fetch``).

Hermes-free by design: the Hermes provider (Phase 2) and a future MCP server both wrap this.
Use as ``async with DeepSeekClient(key) as client`` — an ``httpx.AsyncClient`` is bound to the
event loop that opened it, so do not share one instance across loops.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from hermes_deepseek_search import __version__
from hermes_deepseek_search.core.errors import (
    DeepSeekError,
    MissingApiKeyError,
    NetworkError,
    ResponseParseError,
    error_from_response,
)
from hermes_deepseek_search.core.models import FetchResult, SearchResponse, SearchResult
from hermes_deepseek_search.core.validation import (
    clamp_count,
    validate_base_url,
    validate_localization,
    validate_ori_question,
    validate_query,
    validate_url,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"
_SEARCH_PATH = "/v1/tools/search"
_FETCH_PATH = "/v1/tools/fetch"
_SEARCH_LABEL = "DeepSeek Search"
_FETCH_LABEL = "DeepSeek Fetch"
_MAX_RETRY_AFTER = 30.0


def backoff_delay(
    attempt: int,
    *,
    base: float,
    cap: float,
    retry_after: float | None = None,
    rand: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before retry ``attempt`` (0-based): a server ``Retry-After`` wins (capped at 30s),
    else equal-jitter exponential backoff — half fixed, half random — so 429s never re-fire at ~0s."""
    if retry_after is not None:
        return min(retry_after, _MAX_RETRY_AFTER)
    ceiling = min(cap, base * (2**attempt))
    return ceiling / 2 + rand() * ceiling / 2


class DeepSeekClient:
    """Async DeepSeek Search & Fetch client. Raises :class:`DeepSeekError` subclasses, never returns error dicts."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        search_timeout: float = 30.0,
        fetch_timeout: float = 60.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not api_key or not api_key.strip():
            raise MissingApiKeyError(f"{API_KEY_ENV} is not set.")
        self._api_key = api_key.strip()
        self._base_url = validate_base_url(base_url)
        self._search_timeout = search_timeout
        self._fetch_timeout = fetch_timeout
        self._max_retries = max(0, max_retries)
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._transport = transport
        self._sleep = sleep
        self._http: httpx.AsyncClient | None = None
        self._closed = False

    @classmethod
    def from_env(cls, env_var: str = API_KEY_ENV, **kwargs: Any) -> DeepSeekClient:
        return cls(os.getenv(env_var, ""), **kwargs)

    def __repr__(self) -> str:
        return f"DeepSeekClient(base_url={self._base_url!r}, api_key='***')"

    async def __aenter__(self) -> DeepSeekClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client. Terminal: later requests raise instead of silently reopening a connection."""
        self._closed = True
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("DeepSeekClient is closed")
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "User-Agent": f"hermes-deepseek-search/{__version__}",
                },
                transport=self._transport,
            )
        return self._http

    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        need_content: bool = False,
        ori_question: str | None = None,
        localization: str | None = None,
    ) -> SearchResponse:
        q, truncated = validate_query(query)
        payload: dict[str, Any] = {"query": q, "count": clamp_count(count), "need_content": bool(need_content)}
        if (oq := validate_ori_question(ori_question)) is not None:
            payload["ori_question"] = oq
        if (loc := validate_localization(localization)) is not None:
            payload["localization"] = loc
        data = await self._post(_SEARCH_PATH, payload, label=_SEARCH_LABEL, request_timeout=self._search_timeout)
        raw = data.get("results") if isinstance(data, dict) else None
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise ResponseParseError(f"{_SEARCH_LABEL} response has no 'results' list.")
        results = [SearchResult.from_api(item) for item in raw if isinstance(item, dict) and item.get("url")]
        logger.debug(
            "DeepSeek Search '%s': %d results (from %d raw, count %d%s)",
            q,
            len(results),
            len(raw),
            payload["count"],
            ", query truncated" if truncated else "",
        )
        return SearchResponse(query=q, query_truncated=truncated, results=results)

    async def fetch(self, url: str) -> FetchResult:
        u = validate_url(url)
        data = await self._post(_FETCH_PATH, {"url": u}, label=_FETCH_LABEL, request_timeout=self._fetch_timeout)
        if not isinstance(data, dict):
            raise ResponseParseError(f"{_FETCH_LABEL} response is not a JSON object.")
        page = FetchResult(url=u, title=str(data.get("title") or ""), content=str(data.get("content") or ""))
        logger.debug("DeepSeek Fetch '%s': %d chars", u, len(page.content))
        return page

    async def _post(self, path: str, payload: dict[str, Any], *, label: str, request_timeout: float) -> Any:
        attempt = 0
        while True:
            err: DeepSeekError
            try:
                resp = await self._client().post(path, json=payload, timeout=request_timeout)
            except httpx.DecodingError as exc:  # a RequestError, not a TransportError: retrying will not help
                raise ResponseParseError(f"Could not decode {label} response: {exc}") from exc
            except httpx.TransportError as exc:
                err = NetworkError(f"Could not reach {label}: {exc}")
                err.__cause__ = exc
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise ResponseParseError(f"Could not parse {label} response as JSON") from exc
                err = error_from_response(resp.status_code, resp.text, resp.headers, label=label)
            if not err.retryable or attempt >= self._max_retries:
                if err.retryable:
                    logger.warning("%s giving up after %d attempt(s): %s", label, attempt + 1, err)
                raise err
            delay = backoff_delay(
                attempt,
                base=self._backoff_base,
                cap=self._backoff_cap,
                retry_after=getattr(err, "retry_after", None),
            )
            logger.warning(
                "%s attempt %d/%d failed (%s); retrying in %.2fs",
                label,
                attempt + 1,
                self._max_retries + 1,
                err,
                delay,
            )
            await self._sleep(delay)
            attempt += 1
