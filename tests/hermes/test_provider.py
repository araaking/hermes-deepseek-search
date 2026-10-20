"""Tests for DeepSeekWebSearchProvider, its settings, run_sync and register().

Covers:
- Availability & metadata — env gating, name/display_name, capability flags, setup schema, ABC isinstance
- Dispatch contract — search() is sync, extract() is a coroutine function
- search() — envelope, limit → count mapping, empty results, min_rerank_score filter, truncation notice,
  missing key, API/validation/network/unexpected errors, called while a loop runs, settings wiring
- extract() — order, empty list, mixed success/404/403, invalid URL, missing key, ignored kwargs, duplicates,
  concurrency cap, batch deadline, per-URL and factory failures
- run_sync — with and without a running loop, exception propagation
- ProviderSettings.from_ctx — valid, numeric strings, invalid values (with warnings), ctx without get_config
- register() — registers one provider with ctx settings; entry point metadata; discovery/admission source checks
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import json
import re
from pathlib import Path

import httpx
import pytest

from hermes_deepseek_search.core import DEFAULT_BASE_URL, DeepSeekClient
from hermes_deepseek_search.hermes import register
from hermes_deepseek_search.hermes._sync import run_sync
from hermes_deepseek_search.hermes.provider import DeepSeekWebSearchProvider
from hermes_deepseek_search.hermes.settings import ProviderSettings
from tests.hermes.conftest import FakeContext

_HERMES_PKG = Path(__file__).resolve().parents[2] / "src" / "hermes_deepseek_search" / "hermes"

_ITEM = {
    "rerank_score": 2.4,
    "title": "DeepSeek V4 model released",
    "url": "https://example.com/deepseek-v4",
    "snippet": "DeepSeek V4 is now available.",
    "summary": "DeepSeek has released the V4 model.",
    "content": "",
    "site_name": "Example News",
    "site_icon": None,
    "date": "2026-06-01T00:00:00+00:00",
    "authority": 3,
}


def _ok(data) -> httpx.Response:
    return httpx.Response(200, json=data)


def _error(status: int, message: str = "boom") -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "x", "param": None, "code": "x"}})


def _fetch_by_url(pages: dict[str, httpx.Response]):
    """Reply per requested URL (Fetch body is ``{"url": ...}``)."""

    def reply(request: httpx.Request) -> httpx.Response:
        return pages[json.loads(request.content)["url"]]

    return reply


# ---------------------------------------------------------------------------
# Availability & metadata
# ---------------------------------------------------------------------------


class TestAvailabilityAndMetadata:
    def test_available_when_key_set(self, api_key):
        assert DeepSeekWebSearchProvider().is_available() is True

    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert DeepSeekWebSearchProvider().is_available() is False

    def test_unavailable_with_whitespace_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
        assert DeepSeekWebSearchProvider().is_available() is False

    def test_names_and_capabilities(self):
        p = DeepSeekWebSearchProvider()
        assert p.name == "deepseek"
        assert p.display_name == "DeepSeek Search"
        assert p.supports_search() is True
        assert p.supports_extract() is True
        assert p.is_keyless_available() is False

    def test_setup_schema(self):
        schema = DeepSeekWebSearchProvider().get_setup_schema()
        assert schema["name"] == "DeepSeek Search"
        assert schema["badge"] == "paid"
        assert schema["env_vars"][0]["key"] == "DEEPSEEK_API_KEY"
        assert schema["env_vars"][0]["url"].startswith("https://platform.deepseek.com")

    def test_is_hermes_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider

        assert isinstance(DeepSeekWebSearchProvider(), WebSearchProvider)


class TestDispatchContract:
    def test_search_is_sync(self):
        assert not inspect.iscoroutinefunction(DeepSeekWebSearchProvider.search)

    def test_extract_is_coroutine_function(self):
        assert inspect.iscoroutinefunction(DeepSeekWebSearchProvider().extract)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    def test_happy_path_envelope(self, make_provider, api_key, recorder):
        provider = make_provider([_ok({"results": [_ITEM]})])
        result = provider.search("DeepSeek V4", limit=5)
        assert result == {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "DeepSeek V4 model released",
                        "url": "https://example.com/deepseek-v4",
                        "description": "DeepSeek has released the V4 model.\n"
                        "[Example News · 2026-06-01 · relevance 2.4/3 · authority 3/4]",
                        "position": 1,
                    }
                ]
            },
        }
        assert list(result) == ["success", "data"]
        assert recorder.body() == {"query": "DeepSeek V4", "count": 5, "need_content": False}
        assert recorder.requests[0].headers["authorization"] == "Bearer sk-test"

    @pytest.mark.parametrize(("limit", "count"), [(5, 5), (100, 50), (0, 1), ("abc", 5), (None, 5)])
    def test_limit_maps_to_count(self, make_provider, api_key, recorder, limit, count):
        make_provider([_ok({"results": []})]).search("q", limit=limit)
        assert recorder.body()["count"] == count

    def test_empty_results(self, make_provider, api_key):
        assert make_provider([_ok({"results": []})]).search("q") == {"success": True, "data": {"web": []}}

    def test_min_rerank_score_filter(self, make_provider, api_key):
        items = [
            {**_ITEM, "url": "https://a.example", "rerank_score": 2.4},
            {**_ITEM, "url": "https://b.example", "rerank_score": 1.0},
            {**_ITEM, "url": "https://c.example", "rerank_score": 1.5},
        ]
        provider = make_provider([_ok({"results": items})], ProviderSettings(min_rerank_score=1.5))
        web = provider.search("q")["data"]["web"]
        assert [(r["url"], r["position"]) for r in web] == [("https://a.example", 1), ("https://c.example", 2)]

    def test_truncated_query_adds_notice(self, make_provider, api_key, recorder):
        result = make_provider([_ok({"results": []})]).search(" ".join(["word"] * 50))
        assert list(result) == ["success", "data", "notice"]
        assert recorder.body()["query"] in result["notice"]
        assert "shortened" in result["notice"]

    def test_missing_key(self, make_provider, monkeypatch, recorder):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        result = make_provider([_ok({"results": []})]).search("q")
        assert result == {"success": False, "error": "DEEPSEEK_API_KEY is not set"}
        assert recorder.requests == []

    def test_auth_error_verbatim_without_key(self, make_provider, api_key):
        result = make_provider([_error(401, "bad key")]).search("q")
        assert result["success"] is False
        assert result["error"].startswith("DeepSeek Search returned HTTP 401: bad key")
        assert "sk-test" not in result["error"]

    def test_balance_error_hint(self, make_provider, api_key):
        assert "top up" in make_provider([_error(402, "Insufficient Balance")]).search("q")["error"]

    def test_empty_query_validation_error_sends_nothing(self, make_provider, api_key, recorder):
        result = make_provider([_ok({"results": []})]).search("   ")
        assert result["success"] is False
        assert "must not be empty" in result["error"]
        assert recorder.requests == []

    def test_network_error(self, make_provider, api_key, recorder):
        result = make_provider([httpx.ConnectError("down")]).search("q")
        assert result["success"] is False
        assert "Could not reach DeepSeek Search" in result["error"]
        assert len(recorder.requests) == 3

    def test_unexpected_exception_wrapped(self, api_key):
        def broken_factory(key: str) -> DeepSeekClient:
            raise RuntimeError("boom")

        provider = DeepSeekWebSearchProvider(client_factory=broken_factory)
        assert provider.search("q") == {"success": False, "error": "DeepSeek search failed: boom"}

    async def test_search_called_while_loop_running(self, make_provider, api_key):
        result = make_provider([_ok({"results": [_ITEM]})]).search("q")
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1

    def test_max_retries_from_settings(self, make_provider, api_key, recorder):
        provider = make_provider([_error(503)], ProviderSettings(max_retries=0))
        assert provider.search("q")["success"] is False
        assert len(recorder.requests) == 1

    def test_base_url_from_settings(self, make_provider, api_key, recorder):
        make_provider([_ok({"results": []})], ProviderSettings(base_url="https://proxy.example")).search("q")
        assert str(recorder.requests[0].url) == "https://proxy.example/v1/tools/search"

    def test_default_client_uses_settings(self):
        provider = DeepSeekWebSearchProvider(ProviderSettings(base_url="https://proxy.example", max_retries=4))
        client = provider._default_client("sk-x")
        assert client._base_url == "https://proxy.example"
        assert client._max_retries == 4


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------


class TestExtract:
    async def test_happy_path_preserves_order(self, make_provider, api_key):
        pages = {
            "https://a.example": _ok({"title": "A", "content": "# A"}),
            "https://b.example": _ok({"title": "B", "content": "# B"}),
        }
        out = await make_provider(_fetch_by_url(pages)).extract(["https://b.example", "https://a.example"])
        assert [d["url"] for d in out] == ["https://b.example", "https://a.example"]
        assert out[0] == {
            "url": "https://b.example",
            "title": "B",
            "content": "# B",
            "raw_content": "# B",
            "metadata": {"sourceURL": "https://b.example", "title": "B"},
        }

    async def test_empty_list(self, make_provider, api_key, recorder):
        assert await make_provider([_ok({})]).extract([]) == []
        assert recorder.requests == []

    async def test_mixed_success_404_and_403(self, make_provider, api_key):
        pages = {
            "https://a.example": _ok({"title": "A", "content": "# A"}),
            "https://b.example": _error(404, "no content"),
            "https://c.example": _error(403, "blocked"),
        }
        out = await make_provider(_fetch_by_url(pages)).extract(
            ["https://a.example", "https://b.example", "https://c.example"]
        )
        assert out[0]["content"] == "# A"
        assert out[1] == {
            "url": "https://b.example",
            "title": "",
            "content": "",
            "error": out[1]["error"],
        }
        assert out[1]["error"].startswith("DeepSeek Fetch returned HTTP 404: no content")
        assert "security policy" in out[2]["error"]

    async def test_invalid_url_is_per_entry_error(self, make_provider, api_key, recorder):
        pages = {"https://a.example": _ok({"title": "A", "content": "# A"})}
        out = await make_provider(_fetch_by_url(pages)).extract(["ftp://x.example", "https://a.example"])
        assert "http/https" in out[0]["error"]
        assert out[1]["content"] == "# A"
        assert len(recorder.requests) == 1

    async def test_missing_key(self, make_provider, monkeypatch, recorder):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        out = await make_provider([_ok({})]).extract(["https://a.example", "https://b.example"])
        assert [e["error"] for e in out] == ["DEEPSEEK_API_KEY is not set"] * 2
        assert recorder.requests == []

    async def test_unknown_kwargs_ignored(self, make_provider, api_key):
        out = await make_provider([_ok({"title": "A", "content": "x"})]).extract(
            ["https://a.example"], format="html", include_raw=True, max_chars=10
        )
        assert out[0]["content"] == "x"

    async def test_duplicate_urls_kept(self, make_provider, api_key, recorder):
        out = await make_provider([_ok({"title": "A", "content": "x"})]).extract(
            ["https://a.example", "https://a.example"]
        )
        assert len(out) == 2
        assert len(recorder.requests) == 2

    async def test_concurrency_capped(self, api_key):
        state = {"now": 0, "peak": 0}

        class CountingTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
                await asyncio.sleep(0.02)
                state["now"] -= 1
                return httpx.Response(200, json={"title": "T", "content": "ok"})

        provider = DeepSeekWebSearchProvider(
            ProviderSettings(extract_concurrency=2),
            client_factory=lambda k: DeepSeekClient(k, transport=CountingTransport()),
        )
        out = await provider.extract([f"https://{i}.example" for i in range(6)])
        assert all(e["content"] == "ok" for e in out)
        assert state["peak"] == 2

    async def test_deadline_marks_unfinished_urls(self, api_key):
        class SlowForSome(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if b"slow" in request.content:
                    await asyncio.sleep(5)
                return httpx.Response(200, json={"title": "T", "content": "ok"})

        provider = DeepSeekWebSearchProvider(
            client_factory=lambda k: DeepSeekClient(k, transport=SlowForSome()), extract_deadline=0.2
        )
        out = await provider.extract(["https://a.example", "https://slow.example"])
        assert out[0]["content"] == "ok"
        assert out[1]["error"] == "DeepSeek Fetch did not finish within 0.2s"
        assert out[1]["url"] == "https://slow.example"

    async def test_unexpected_per_url_exception(self, api_key):
        class Exploding(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if b"bad" in request.content:
                    raise RuntimeError("kaboom")
                return httpx.Response(200, json={"title": "T", "content": "ok"})

        provider = DeepSeekWebSearchProvider(client_factory=lambda k: DeepSeekClient(k, transport=Exploding()))
        out = await provider.extract(["https://ok.example", "https://bad.example"])
        assert out[0]["content"] == "ok"
        assert out[1]["error"] == "DeepSeek extract failed: kaboom"

    async def test_client_factory_failure(self, api_key):
        def broken_factory(key: str) -> DeepSeekClient:
            raise RuntimeError("no client")

        out = await DeepSeekWebSearchProvider(client_factory=broken_factory).extract(["https://a.example"])
        assert out == [
            {"url": "https://a.example", "title": "", "content": "", "error": "DeepSeek extract failed: no client"}
        ]


# ---------------------------------------------------------------------------
# run_sync
# ---------------------------------------------------------------------------


async def _value(v):
    return v


async def _raise():
    raise ValueError("nope")


class TestRunSync:
    def test_without_running_loop(self):
        assert run_sync(lambda: _value(1)) == 1

    async def test_with_running_loop(self):
        assert run_sync(lambda: _value(2)) == 2

    def test_exception_propagates(self):
        with pytest.raises(ValueError, match="nope"):
            run_sync(_raise)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self):
        assert ProviderSettings() == ProviderSettings(0.0, 2, DEFAULT_BASE_URL, 5)

    def test_from_ctx_valid(self):
        ctx = FakeContext(
            {"min_rerank_score": 1.5, "max_retries": 0, "base_url": "https://p.example", "extract_concurrency": 10}
        )
        assert ProviderSettings.from_ctx(ctx) == ProviderSettings(1.5, 0, "https://p.example", 10)

    def test_from_ctx_numeric_strings(self):
        s = ProviderSettings.from_ctx(FakeContext({"max_retries": "3", "min_rerank_score": "1.2"}))
        assert s.max_retries == 3
        assert s.min_rerank_score == 1.2

    @pytest.mark.parametrize(
        ("settings", "field"),
        [
            ({"max_retries": "x"}, "max_retries"),
            ({"max_retries": True}, "max_retries"),
            ({"max_retries": 9}, "max_retries"),
            ({"min_rerank_score": 9}, "min_rerank_score"),
            ({"min_rerank_score": None}, "min_rerank_score"),
            ({"extract_concurrency": 0}, "extract_concurrency"),
            ({"extract_concurrency": 100}, "extract_concurrency"),
            ({"base_url": "ftp://x"}, "base_url"),
            ({"base_url": 42}, "base_url"),
        ],
    )
    def test_from_ctx_invalid_falls_back_with_warning(self, caplog, settings, field):
        with caplog.at_level("WARNING"):
            s = ProviderSettings.from_ctx(FakeContext(settings))
        assert getattr(s, field) == getattr(ProviderSettings(), field)
        assert f"web-deepseek setting {field}=" in caplog.text

    def test_ctx_without_get_config(self):
        assert ProviderSettings.from_ctx(object()) == ProviderSettings()


# ---------------------------------------------------------------------------
# register() & packaging
# ---------------------------------------------------------------------------


class TestRegister:
    def test_registers_one_provider(self):
        ctx = FakeContext()
        register(ctx)
        assert len(ctx.providers) == 1
        assert isinstance(ctx.providers[0], DeepSeekWebSearchProvider)
        assert ctx.providers[0].name == "deepseek"

    def test_uses_ctx_settings(self):
        ctx = FakeContext({"extract_concurrency": 3})
        register(ctx)
        assert ctx.providers[0]._settings.extract_concurrency == 3

    def test_entry_point_declared(self):
        eps = importlib.metadata.entry_points(group="hermes_agent.plugins")
        assert [(e.name, e.value) for e in eps if e.name == "web-deepseek"] == [
            ("web-deepseek", "hermes_deepseek_search.hermes")
        ]

    def test_entry_module_free_of_exclusive_kind_markers(self):
        source = (_HERMES_PKG / "__init__.py").read_text(encoding="utf-8")
        for marker in (
            "MemoryProvider",
            "register_memory_provider",
            "CronScheduler",
            "ComputerUseProvider",
            "ProviderProfile",
        ):
            assert marker not in source

    def test_only_public_hermes_imports(self):
        hermes_roots = ("agent", "plugins", "tools", "hermes_cli", "gateway")
        found = set()
        for path in _HERMES_PKG.glob("*.py"):
            for match in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", path.read_text(encoding="utf-8"), re.M):
                if match.group(1).split(".")[0] in hermes_roots:
                    found.add(match.group(1))
        assert found == {"agent.web_search_provider"}
