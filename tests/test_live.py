"""Live canaries against the real DeepSeek API (spends ~$0.0015; run with: uv run pytest -m live)."""

from __future__ import annotations

import os

import pytest

from hermes_deepseek_search.core import DeepSeekClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"),
]


class TestLive:
    async def test_search_returns_results(self):
        async with DeepSeekClient.from_env() as client:
            resp = await client.search("DeepSeek API", count=3)
        assert resp.results
        assert resp.results[0].url.startswith("http")

    async def test_fetch_returns_markdown(self):
        async with DeepSeekClient.from_env() as client:
            page = await client.fetch("https://example.com")
        assert page.content
