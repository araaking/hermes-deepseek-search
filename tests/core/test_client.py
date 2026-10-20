"""Tests for DeepSeekClient against httpx.MockTransport.

Covers:
- Construction — missing key, from_env, repr never leaks the key, aclose idempotent
- Request shape — endpoint URLs, method, headers, JSON payloads (optional fields omitted), custom base_url
- search() parsing — happy path, empty/missing/null results, non-list results, junk items, null/mistyped fields
- fetch() parsing — happy path, null title, non-object body
- Errors — non-JSON 200, typed API errors, key never in messages
- Retry — 429 honoring Retry-After, 5xx exhaustion, no retry for 4xx, network errors, max_retries=0
- backoff_delay() — equal-jitter bounds, cap, Retry-After cap
- Local validation sends no request
"""

from __future__ import annotations

import httpx
import pytest

from hermes_deepseek_search.core import (
    AuthenticationError,
    ContentNotFoundError,
    DeepSeekClient,
    FetchBlockedError,
    FetchResult,
    InsufficientBalanceError,
    InvalidRequestError,
    MissingApiKeyError,
    NetworkError,
    ResponseParseError,
    SearchResult,
    ServerError,
    ValidationError,
)
from hermes_deepseek_search.core.client import backoff_delay

_FULL_ITEM = {
    "rerank_score": 2.4,
    "title": "DeepSeek V4 model released",
    "url": "https://example.com/deepseek-v4",
    "snippet": "DeepSeek V4 is now available.",
    "summary": "DeepSeek has released the V4 model.",
    "content": "",
    "site_name": "Example News",
    "site_icon": "https://example.com/favicon.ico",
    "date": "2026-06-01T00:00:00+00:00",
    "authority": 3,
}


def _ok(json_data) -> httpx.Response:
    return httpx.Response(200, json=json_data)


def _error(status: int, message: str = "boom", headers: dict[str, str] | None = None) -> httpx.Response:
    body = {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": "x"}}
    return httpx.Response(status, json=body, headers=headers)


# ---------------------------------------------------------------------------
# Construction & lifecycle
# ---------------------------------------------------------------------------


class TestConstruction:
    @pytest.mark.parametrize("key", ["", "   "])
    def test_missing_key_raises(self, key):
        with pytest.raises(MissingApiKeyError, match="DEEPSEEK_API_KEY"):
            DeepSeekClient(key)

    def test_from_env_missing_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(MissingApiKeyError):
            DeepSeekClient.from_env()

    async def test_from_env_uses_env_key(self, monkeypatch, recorder):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            return _ok({"results": []})

        async with DeepSeekClient.from_env(transport=httpx.MockTransport(handler)) as client:
            await client.search("q")
        assert recorder.requests[0].headers["authorization"] == "Bearer sk-env"

    def test_repr_hides_key(self, make_client):
        client = make_client(_ok({}))
        assert "sk-test" not in repr(client)
        assert "***" in repr(client)

    async def test_aclose_idempotent(self, make_client):
        client = make_client(_ok({"results": []}))
        await client.aclose()
        await client.search("q")
        await client.aclose()
        await client.aclose()


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    async def test_search_endpoint_headers_and_minimal_payload(self, make_client, recorder):
        async with make_client(_ok({"results": []})) as client:
            await client.search("q")
        req = recorder.requests[0]
        assert req.method == "POST"
        assert str(req.url) == "https://api.deepseek.com/v1/tools/search"
        assert req.headers["authorization"] == "Bearer sk-test"
        assert req.headers["content-type"] == "application/json"
        assert req.headers["accept"] == "application/json"
        assert req.headers["user-agent"].startswith("hermes-deepseek-search/")
        assert recorder.body() == {"query": "q", "count": 10, "need_content": False}

    async def test_search_full_payload(self, make_client, recorder):
        async with make_client(_ok({"results": []})) as client:
            await client.search("q", count=5, need_content=True, ori_question="why?", localization="1.1.1.1")
        assert recorder.body() == {
            "query": "q",
            "count": 5,
            "need_content": True,
            "ori_question": "why?",
            "localization": "1.1.1.1",
        }

    async def test_search_count_clamped(self, make_client, recorder):
        async with make_client(_ok({"results": []})) as client:
            await client.search("q", count=500)
        assert recorder.body()["count"] == 50

    async def test_search_truncated_query_flagged(self, make_client, recorder):
        async with make_client(_ok({"results": []})) as client:
            resp = await client.search(" ".join(["word"] * 50))
        assert resp.query_truncated is True
        assert len(recorder.body()["query"]) <= 128
        assert resp.query == recorder.body()["query"]

    async def test_fetch_endpoint_and_payload(self, make_client, recorder):
        async with make_client(_ok({"title": "T", "content": "# T"})) as client:
            await client.fetch(" https://example.com/article ")
        req = recorder.requests[0]
        assert str(req.url) == "https://api.deepseek.com/v1/tools/fetch"
        assert recorder.body() == {"url": "https://example.com/article"}

    async def test_custom_base_url_trailing_slash_stripped(self, make_client, recorder):
        async with make_client(_ok({"results": []}), base_url="https://proxy.example/") as client:
            await client.search("q")
        assert str(recorder.requests[0].url) == "https://proxy.example/v1/tools/search"


# ---------------------------------------------------------------------------
# search() parsing
# ---------------------------------------------------------------------------


class TestSearchParsing:
    async def test_happy_path(self, make_client):
        second = {**_FULL_ITEM, "url": "https://b.example.com", "title": "B"}
        async with make_client(_ok({"results": [_FULL_ITEM, second]})) as client:
            resp = await client.search("DeepSeek V4 release")
        assert resp.query == "DeepSeek V4 release"
        assert resp.query_truncated is False
        assert len(resp.results) == 2
        assert resp.results[0] == SearchResult(
            title="DeepSeek V4 model released",
            url="https://example.com/deepseek-v4",
            snippet="DeepSeek V4 is now available.",
            summary="DeepSeek has released the V4 model.",
            content="",
            rerank_score=2.4,
            site_name="Example News",
            site_icon="https://example.com/favicon.ico",
            date="2026-06-01T00:00:00+00:00",
            authority=3,
        )
        assert resp.results[1].title == "B"

    @pytest.mark.parametrize("payload", [{"results": []}, {}, {"results": None}, []])
    async def test_empty_or_missing_results(self, make_client, payload):
        async with make_client(_ok(payload)) as client:
            resp = await client.search("q")
        assert resp.results == []

    async def test_non_list_results_raises(self, make_client):
        async with make_client(_ok({"results": "x"})) as client:
            with pytest.raises(ResponseParseError, match="results"):
                await client.search("q")

    async def test_junk_items_skipped(self, make_client):
        items = [{"title": "no url"}, "str", None, {"title": "ok", "url": "https://ok.example"}]
        async with make_client(_ok({"results": items})) as client:
            resp = await client.search("q")
        assert [r.url for r in resp.results] == ["https://ok.example"]

    async def test_null_and_mistyped_fields_tolerated(self, make_client):
        item = {
            "url": "https://x.example",
            "title": None,
            "rerank_score": None,
            "authority": "x",
            "site_name": None,
            "site_icon": "",
            "date": None,
        }
        async with make_client(_ok({"results": [item]})) as client:
            resp = await client.search("q")
        r = resp.results[0]
        assert r.title == ""
        assert r.rerank_score == 0.0
        assert r.authority == -1
        assert r.site_name is None
        assert r.site_icon is None
        assert r.date is None

    async def test_non_numeric_score_and_null_authority(self, make_client):
        item = {"url": "https://x.example", "rerank_score": "high", "authority": None}
        async with make_client(_ok({"results": [item]})) as client:
            resp = await client.search("q")
        assert resp.results[0].rerank_score == 0.0
        assert resp.results[0].authority == -1


# ---------------------------------------------------------------------------
# fetch() parsing
# ---------------------------------------------------------------------------


class TestFetchParsing:
    async def test_happy_path(self, make_client):
        async with make_client(_ok({"title": "Example article", "content": "# Example article\n\nBody."})) as client:
            page = await client.fetch("https://example.com/article")
        assert page == FetchResult(
            url="https://example.com/article", title="Example article", content="# Example article\n\nBody."
        )

    async def test_null_title(self, make_client):
        async with make_client(_ok({"title": None, "content": "x"})) as client:
            page = await client.fetch("https://example.com")
        assert page.title == ""
        assert page.content == "x"

    async def test_non_object_body_raises(self, make_client):
        async with make_client(_ok([])) as client:
            with pytest.raises(ResponseParseError, match="JSON object"):
                await client.fetch("https://example.com")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    async def test_non_json_200_raises_parse_error(self, make_client):
        async with make_client(httpx.Response(200, text="<html>")) as client:
            with pytest.raises(ResponseParseError, match="Could not parse DeepSeek Search response as JSON"):
                await client.search("q")

    @pytest.mark.parametrize(
        ("status", "cls"),
        [
            (400, InvalidRequestError),
            (401, AuthenticationError),
            (402, InsufficientBalanceError),
            (403, FetchBlockedError),
            (404, ContentNotFoundError),
        ],
    )
    async def test_non_retryable_errors_raise_once(self, make_client, recorder, status, cls):
        async with make_client(_error(status)) as client:
            with pytest.raises(cls):
                await client.search("q")
        assert len(recorder.requests) == 1
        assert recorder.sleeps == []

    async def test_fetch_404_message(self, make_client):
        async with make_client(_error(404, "no content")) as client:
            with pytest.raises(ContentNotFoundError, match="DeepSeek Fetch returned HTTP 404: no content"):
                await client.fetch("https://example.com/missing")

    async def test_key_never_in_error_message(self, make_client):
        async with make_client(_error(401, "bad key")) as client:
            with pytest.raises(AuthenticationError) as excinfo:
                await client.search("q")
        assert "sk-test" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


class TestRetry:
    async def test_retries_429_then_succeeds_honoring_retry_after(self, make_client, recorder):
        client = make_client(_error(429, "rate limited", {"Retry-After": "2"}), _ok({"results": []}))
        async with client:
            resp = await client.search("q")
        assert resp.results == []
        assert len(recorder.requests) == 2
        assert recorder.sleeps == [2.0]

    async def test_server_error_exhausts_retries(self, make_client, recorder):
        async with make_client(_error(500), max_retries=2) as client:
            with pytest.raises(ServerError):
                await client.search("q")
        assert len(recorder.requests) == 3
        assert len(recorder.sleeps) == 2

    async def test_retries_408(self, make_client, recorder):
        async with make_client(_error(408), _ok({"title": "", "content": "x"})) as client:
            page = await client.fetch("https://example.com")
        assert page.content == "x"
        assert len(recorder.requests) == 2

    async def test_network_error_then_success(self, make_client, recorder):
        async with make_client(httpx.ConnectError("boom"), _ok({"results": []})) as client:
            await client.search("q")
        assert len(recorder.requests) == 2

    async def test_network_error_exhausted(self, make_client, recorder):
        async with make_client(httpx.ReadTimeout("slow")) as client:
            with pytest.raises(NetworkError, match="Could not reach DeepSeek Search") as excinfo:
                await client.search("q")
        assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)
        assert len(recorder.requests) == 3

    async def test_max_retries_zero(self, make_client, recorder):
        async with make_client(_error(503), max_retries=0) as client:
            with pytest.raises(ServerError):
                await client.search("q")
        assert len(recorder.requests) == 1
        assert recorder.sleeps == []

    async def test_backoff_sleeps_within_bounds(self, make_client, recorder):
        async with make_client(_error(502), max_retries=2, backoff_base=0.5, backoff_cap=8.0) as client:
            with pytest.raises(ServerError):
                await client.search("q")
        assert 0.25 <= recorder.sleeps[0] <= 0.5
        assert 0.5 <= recorder.sleeps[1] <= 1.0


class TestBackoffDelay:
    @pytest.mark.parametrize(
        ("attempt", "low", "high"),
        [(0, 0.25, 0.5), (1, 0.5, 1.0), (2, 1.0, 2.0), (3, 2.0, 4.0), (4, 4.0, 8.0), (5, 4.0, 8.0), (10, 4.0, 8.0)],
    )
    def test_equal_jitter_bounds_and_cap(self, attempt, low, high):
        assert backoff_delay(attempt, base=0.5, cap=8.0, rand=lambda: 0.0) == low
        assert backoff_delay(attempt, base=0.5, cap=8.0, rand=lambda: 1.0) == high

    def test_retry_after_wins(self):
        assert backoff_delay(0, base=0.5, cap=8.0, retry_after=2.0) == 2.0

    def test_retry_after_capped(self):
        assert backoff_delay(0, base=0.5, cap=8.0, retry_after=100.0) == 30.0


# ---------------------------------------------------------------------------
# Validation happens before any request (no cost)
# ---------------------------------------------------------------------------


class TestValidationSendsNothing:
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.search(""),
            lambda c: c.search("q", localization="not-ip"),
            lambda c: c.search("q", count="abc"),
            lambda c: c.fetch("ftp://x.com"),
            lambda c: c.fetch("https://user:pw@x.com"),
        ],
    )
    async def test_invalid_input_sends_no_request(self, make_client, recorder, call):
        async with make_client(_ok({})) as client:
            with pytest.raises(ValidationError):
                await call(client)
        assert recorder.requests == []
