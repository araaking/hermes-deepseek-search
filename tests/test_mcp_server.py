"""Tests for the MCP server, driven through the SDK's in-memory ``Client``.

Covers:
- Tool listing — names, titles, read-only annotations, input schema bounds
- deepseek_search — formatted results, count / ori_question passthrough, out-of-range count, long-query notice,
  empty results, missing key, API / validation / unexpected errors, overall deadline
- deepseek_fetch — cleaned Markdown, truncation, empty page, invalid URL, 404, deadline, unexpected errors
- format_search / format_page — exact text
- main() — --version, missing ``mcp`` extra, other import errors, stdio run
"""

from __future__ import annotations

import asyncio
import logging
import sys

import httpx
import pytest
from mcp import Client
from mcp.server import MCPServer

from hermes_deepseek_search import __version__, mcp_server
from hermes_deepseek_search.core import DeepSeekClient, FetchResult, SearchResponse, SearchResult
from hermes_deepseek_search.mcp_server import build_server, format_page, format_search

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


async def _call(server: MCPServer, tool: str, args: dict):
    async with Client(server) as client:
        return await client.call_tool(tool, args)


def _text(result) -> str:
    return result.content[0].text


class _Slow(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={})


def _broken_factory() -> DeepSeekClient:
    raise RuntimeError("factory boom")


@pytest.fixture
def server(make_client):
    """``server(*responses, **build_kwargs)``: a server whose client serves ``responses`` in order."""

    def factory(*responses: httpx.Response | Exception, **kwargs) -> MCPServer:
        return build_server(lambda: make_client(*responses), **kwargs)

    return factory


class TestTools:
    async def test_listing(self, server):
        async with Client(server(_ok({}))) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == {"deepseek_search", "deepseek_fetch"}
        for tool in tools.values():
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.open_world_hint is True
            assert tool.output_schema is None
        search = tools["deepseek_search"].input_schema["properties"]
        assert (search["count"]["minimum"], search["count"]["maximum"], search["count"]["default"]) == (1, 50, 10)
        assert "maxLength" not in search["query"]  # long queries are shortened by core, not rejected
        assert tools["deepseek_search"].input_schema["required"] == ["query"]
        fetch = tools["deepseek_fetch"].input_schema["properties"]["max_chars"]
        assert (fetch["minimum"], fetch["maximum"], fetch["default"]) == (1_000, 500_000, 50_000)


class TestSearch:
    async def test_results_formatted(self, server, recorder):
        second = {**_ITEM, "title": "Second", "url": "https://b.x"}
        result = await _call(server(_ok({"results": [_ITEM, second]})), "deepseek_search", {"query": "deepseek v4"})
        assert result.is_error is False
        assert result.structured_content is None
        text = _text(result)
        assert text.startswith("Results for 'deepseek v4': 2")
        assert "1. DeepSeek V4 model released\n   https://example.com/deepseek-v4" in text
        assert "[Example News · 2026-06-01 · relevance 2.4/3 · authority 3/4]" in text
        assert "2. Second" in text
        assert recorder.body() == {"query": "deepseek v4", "count": 10, "need_content": False}

    async def test_count_and_ori_question_forwarded(self, server, recorder):
        args = {"query": "q", "count": 3, "ori_question": "why is the sky blue?"}
        await _call(server(_ok({"results": []})), "deepseek_search", args)
        assert recorder.body()["count"] == 3
        assert recorder.body()["ori_question"] == "why is the sky blue?"

    async def test_count_out_of_range_rejected_before_request(self, server, recorder):
        result = await _call(server(_ok({"results": []})), "deepseek_search", {"query": "q", "count": 99})
        assert result.is_error is True
        assert recorder.requests == []

    async def test_long_query_notice(self, server):
        result = await _call(server(_ok({"results": [_ITEM]})), "deepseek_search", {"query": " ".join(["word"] * 40)})
        assert "Query was longer than 128 characters and was shortened to:" in _text(result)

    async def test_empty_results_are_not_an_error(self, server):
        result = await _call(server(_ok({"results": []})), "deepseek_search", {"query": "nothing"})
        assert result.is_error is False
        assert _text(result) == "No results for 'nothing'."

    async def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        result = await _call(build_server(), "deepseek_search", {"query": "q"})
        assert result.is_error is True
        assert "DEEPSEEK_API_KEY is not set." in _text(result)
        assert "-e DEEPSEEK_API_KEY=" in _text(result)

    async def test_api_error(self, server):
        result = await _call(server(_error(401, "Authentication Fails")), "deepseek_search", {"query": "q"})
        assert result.is_error is True
        assert "HTTP 401" in _text(result)

    async def test_blank_query(self, server, recorder):
        result = await _call(server(_ok({})), "deepseek_search", {"query": "   "})
        assert result.is_error is True
        assert "must not be empty" in _text(result)
        assert recorder.requests == []

    async def test_deadline(self):
        slow = build_server(lambda: DeepSeekClient("sk-test", transport=_Slow()), search_deadline=0.2)
        result = await _call(slow, "deepseek_search", {"query": "q"})
        assert "DeepSeek Search did not finish within 0.2s" in _text(result)

    async def test_unexpected_error(self, caplog):
        with caplog.at_level(logging.WARNING, logger="hermes_deepseek_search.mcp_server"):
            result = await _call(build_server(_broken_factory), "deepseek_search", {"query": "q"})
        assert result.is_error is True
        assert "DeepSeek search failed: factory boom" in _text(result)
        assert "factory boom" in caplog.text

    async def test_query_not_logged_at_info(self, server, caplog):
        with caplog.at_level(logging.INFO):
            await _call(server(_ok({"results": []})), "deepseek_search", {"query": "very private question"})
        assert "very private question" not in caplog.text


class TestFetch:
    async def test_page_cleaned(self, server, recorder):
        page = _ok({"title": "LP", "content": "\\# Radiofrequency\n\\- 60 min"})
        result = await _call(server(page), "deepseek_fetch", {"url": "https://a.example/lp"})
        assert result.is_error is False
        assert _text(result) == "# LP\nSource: https://a.example/lp\n\n# Radiofrequency\n- 60 min"
        assert recorder.body() == {"url": "https://a.example/lp"}

    async def test_truncated(self, server):
        args = {"url": "https://a.example", "max_chars": 1_000}
        result = await _call(server(_ok({"title": "T", "content": "x" * 5_000})), "deepseek_fetch", args)
        text = _text(result)
        assert "x" * 1_000 + "\n\n[Truncated: showing 1,000 of 5,000 characters." in text
        assert "x" * 1_001 not in text

    async def test_invalid_url_rejected_before_request(self, server, recorder):
        result = await _call(server(_ok({})), "deepseek_fetch", {"url": "ftp://a.example"})
        assert result.is_error is True
        assert recorder.requests == []

    async def test_not_found(self, server):
        result = await _call(server(_error(404, "no content")), "deepseek_fetch", {"url": "https://a.example"})
        assert result.is_error is True
        assert "HTTP 404" in _text(result)

    async def test_deadline(self):
        slow = build_server(lambda: DeepSeekClient("sk-test", transport=_Slow()), fetch_deadline=0.2)
        result = await _call(slow, "deepseek_fetch", {"url": "https://a.example"})
        assert "DeepSeek Fetch did not finish within 0.2s" in _text(result)

    async def test_unexpected_error(self):
        result = await _call(build_server(_broken_factory), "deepseek_fetch", {"url": "https://a.example"})
        assert "DeepSeek fetch failed: factory boom" in _text(result)


class TestFormatters:
    def test_search_without_metadata_body(self):
        bare = SearchResult.from_api({"title": "T", "url": "https://t.x", "rerank_score": 1.0, "authority": -1})
        assert format_search(SearchResponse(query="q", query_truncated=False, results=[bare])) == (
            "Results for 'q': 1\n\n1. T\n   https://t.x\n   [relevance 1.0/3]"
        )

    def test_search_multiline_description_indented(self):
        item = SearchResult.from_api(_ITEM)
        text = format_search(SearchResponse(query="q", query_truncated=True, results=[item]))
        assert text.splitlines()[1] == "Query was longer than 128 characters and was shortened to: 'q'"
        assert "   DeepSeek has released the V4 model.\n   [Example News" in text

    def test_page_empty_content(self):
        assert format_page(FetchResult(url="https://a.x", title="", content="  "), 1_000) == (
            "# https://a.x\nSource: https://a.x\n\n(no extractable content)"
        )

    def test_page_exactly_max_chars_not_truncated(self):
        text = format_page(FetchResult(url="https://a.x", title="T", content="y" * 1_000), 1_000)
        assert text.endswith("y" * 1_000)
        assert "Truncated" not in text


class TestMain:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as info:
            mcp_server.main(["--version"])
        assert info.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_missing_extra(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "mcp.server", None)
        assert mcp_server.main([]) == 2
        out = capsys.readouterr()
        assert "hermes-deepseek-search[mcp]" in out.err
        assert out.out == ""  # stdout belongs to the protocol

    def test_other_import_error_propagates(self, monkeypatch):
        def broken(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'pydanticx'", name="pydanticx")

        monkeypatch.setattr(mcp_server, "build_server", broken)
        with pytest.raises(ModuleNotFoundError):
            mcp_server.main([])

    def test_runs_stdio(self, monkeypatch):
        runs = []
        monkeypatch.setattr(MCPServer, "run", lambda self, *args, **kwargs: runs.append((self.name, args, kwargs)))
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
        assert mcp_server.main([]) == 0
        assert runs == [("deepseek-search", (), {})]
