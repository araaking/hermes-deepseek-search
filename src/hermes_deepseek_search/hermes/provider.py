"""DeepSeek Search & Fetch as a Hermes web provider — ``web_search`` + ``web_extract``.

Config: ``web.backend`` (or ``web.search_backend`` / ``web.extract_backend``): ``"deepseek"``.
Env: ``DEEPSEEK_API_KEY`` — the same key Hermes uses for DeepSeek models. Billed per successful
request ($0.75 / 1k); failed and empty requests are free.

``search()`` is sync because ``tools.web_tools`` calls it synchronously; ``extract()`` is async (awaited
by the dispatcher under ``web.extract_timeout``). A fresh ``DeepSeekClient`` is opened per call so no
httpx client crosses event loops (gateway threads, ``plugins.isolation: host``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from agent.web_search_provider import WebSearchProvider, get_provider_env

from hermes_deepseek_search.core import API_KEY_ENV, DeepSeekClient, DeepSeekError
from hermes_deepseek_search.hermes._sync import run_sync
from hermes_deepseek_search.hermes.envelope import (
    document,
    page_error,
    search_fail,
    search_ok,
    search_rows,
    setup_schema,
)
from hermes_deepseek_search.hermes.settings import ProviderSettings

logger = logging.getLogger(__name__)

_MAX_SEARCH_COUNT = 50  # DeepSeek caps count at 50; Hermes buckets limits to 10/20/50/100
_EXTRACT_DEADLINE_S = 100.0  # below Hermes' default web.extract_timeout (120s) so finished pages survive
_SEARCH_DEADLINE_S = 60.0  # search() is sync: one call never blocks Hermes longer than this, retries included

ClientFactory = Callable[[str], DeepSeekClient]


class DeepSeekWebSearchProvider(WebSearchProvider):
    """Search + extract provider backed by DeepSeek's ``/v1/tools/search`` and ``/v1/tools/fetch``."""

    NAME = "deepseek"
    DISPLAY_NAME = "DeepSeek Search"
    KEY_ENV = API_KEY_ENV

    def __init__(
        self,
        settings: ProviderSettings | None = None,
        *,
        client_factory: ClientFactory | None = None,
        extract_deadline: float = _EXTRACT_DEADLINE_S,
        search_deadline: float = _SEARCH_DEADLINE_S,
    ) -> None:
        self._settings = settings or ProviderSettings()
        self._client_factory = client_factory or self._default_client
        self._extract_deadline = extract_deadline
        self._search_deadline = search_deadline

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    def is_available(self) -> bool:
        return bool(get_provider_env(self.KEY_ENV))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            self.DISPLAY_NAME,
            "paid",
            "Search + extract via DeepSeek's Search & Fetch API — $0.75 per 1k successful requests; "
            "reuses your DeepSeek model key.",
            self.KEY_ENV,
            "DeepSeek API key",
            "https://platform.deepseek.com/api_keys",
        )

    # --- web_search (sync) -------------------------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        api_key = get_provider_env(self.KEY_ENV)
        if not api_key:
            return search_fail(f"{self.KEY_ENV} is not set")
        try:
            return run_sync(lambda: self._search_async(api_key, query, limit))
        except TimeoutError:
            return search_fail(f"DeepSeek Search did not finish within {self._search_deadline:g}s")
        except DeepSeekError as exc:
            return search_fail(str(exc))
        except Exception as exc:  # never raise into the tool wrapper
            logger.warning("DeepSeek search error: %s", exc)
            return search_fail(f"DeepSeek search failed: {exc}")

    async def _search_async(self, api_key: str, query: str, limit: int) -> dict[str, Any]:
        try:
            count = max(1, min(int(limit), _MAX_SEARCH_COUNT))
        except (TypeError, ValueError):
            count = 5
        async with asyncio.timeout(self._search_deadline):
            async with self._client_factory(api_key) as client:
                resp = await client.search(query, count=count)
        kept = [r for r in resp.results if r.rerank_score >= self._settings.min_rerank_score]
        logger.info("DeepSeek Search: %d results (from %d raw, limit %d)", len(kept), len(resp.results), count)
        notice = (
            f"Query was longer than 128 characters and was shortened to: {resp.query!r}"
            if resp.query_truncated
            else None
        )
        return search_ok(search_rows(kept), notice=notice)

    # --- web_extract (async) -----------------------------------------------------------------------------------

    async def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """One Fetch call per URL, ``extract_concurrency`` at a time, under a batch deadline. Order is preserved;
        per-URL failures become ``error`` entries and never raise. ``format`` etc. are ignored (Markdown only)."""
        urls = list(urls)
        if not urls:
            return []
        api_key = get_provider_env(self.KEY_ENV)
        if not api_key:
            return [page_error(u, f"{self.KEY_ENV} is not set") for u in urls]
        logger.info("DeepSeek extract: %d URL(s)", len(urls))
        try:
            client = self._client_factory(api_key)
        except Exception as exc:  # e.g. invalid settings — still one entry per URL
            logger.warning("DeepSeek extract error: %s", exc)
            return [page_error(u, f"DeepSeek extract failed: {exc}") for u in urls]
        try:
            return await self._fetch_all(client, urls)
        finally:
            try:
                await client.aclose()
            except Exception as exc:  # a failed close must not discard pages that were already fetched
                logger.warning("DeepSeek extract: closing the HTTP client failed: %s", exc)

    async def _fetch_all(self, client: DeepSeekClient, urls: list[str]) -> list[dict[str, Any]]:
        gate = asyncio.Semaphore(self._settings.extract_concurrency)

        async def fetch_one(url: str) -> dict[str, Any]:
            async with gate:
                try:
                    page = await client.fetch(url)
                except DeepSeekError as exc:
                    return page_error(url, str(exc))
                except Exception as exc:
                    logger.warning("DeepSeek extract error for %s: %s", url, exc)
                    return page_error(url, f"DeepSeek extract failed: {exc}")
                return document(page.url, page.title, page.content)

        tasks = [asyncio.create_task(fetch_one(u)) for u in urls]
        try:
            done, pending = await asyncio.wait(tasks, timeout=self._extract_deadline)
        finally:
            # Deadline reached, or Hermes cancelled us (web.extract_timeout): stop every unfinished fetch before
            # the client closes, so nothing keeps running — and billing — in the background.
            unfinished = [t for t in tasks if not t.done()]
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
        if pending:
            logger.warning("DeepSeek extract: %d URL(s) unfinished after %.0fs", len(pending), self._extract_deadline)
        timeout_msg = f"DeepSeek Fetch did not finish within {self._extract_deadline:g}s"
        return [t.result() if t in done else page_error(u, timeout_msg) for t, u in zip(tasks, urls, strict=True)]

    # --- helpers -----------------------------------------------------------------------------------------------

    def _default_client(self, api_key: str) -> DeepSeekClient:
        return DeepSeekClient(api_key, base_url=self._settings.base_url, max_retries=self._settings.max_retries)
