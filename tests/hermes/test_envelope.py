"""Tests for the replicated Hermes result shapes.

Covers:
- search_ok / search_fail — exact dicts and key order, optional ``notice``
- title_hit / search_rows — row key order and 1-based positions
- document / page_error — extract entries matching ``plugins.web._common``
- setup_schema — with and without an env var
- format_description — summary/snippet fallback, optional metadata parts, empty body
"""

from __future__ import annotations

from hermes_deepseek_search.core import SearchResult
from hermes_deepseek_search.hermes.envelope import (
    document,
    format_description,
    page_error,
    search_fail,
    search_ok,
    search_rows,
    setup_schema,
)

_FULL = SearchResult(
    title="DeepSeek V4 model released",
    url="https://example.com/deepseek-v4",
    snippet="Short snippet.",
    summary="DeepSeek has released the V4 model.",
    rerank_score=2.4,
    site_name="Example News",
    date="2026-06-01T00:00:00+00:00",
    authority=3,
)

# ---------------------------------------------------------------------------
# Search envelope
# ---------------------------------------------------------------------------


class TestSearchEnvelope:
    def test_search_ok_without_notice(self):
        row = {"title": "t", "url": "u", "description": "d", "position": 1}
        result = search_ok([row])
        assert result == {"success": True, "data": {"web": [row]}}
        assert list(result) == ["success", "data"]

    def test_search_ok_with_notice_appended_last(self):
        result = search_ok([], notice="shortened")
        assert list(result) == ["success", "data", "notice"]
        assert result["notice"] == "shortened"

    def test_search_fail(self):
        assert search_fail("err") == {"success": False, "error": "err"}

    def test_search_rows_positions_and_key_order(self):
        rows = search_rows([_FULL, _FULL, _FULL])
        assert [r["position"] for r in rows] == [1, 2, 3]
        assert list(rows[0]) == ["title", "url", "description", "position"]
        assert rows[0]["title"] == "DeepSeek V4 model released"


# ---------------------------------------------------------------------------
# Extract entries
# ---------------------------------------------------------------------------


class TestExtractEntries:
    def test_document_shape(self):
        doc = document("https://x.example", "T", "# T")
        assert doc == {
            "url": "https://x.example",
            "title": "T",
            "content": "# T",
            "raw_content": "# T",
            "metadata": {"sourceURL": "https://x.example", "title": "T"},
        }
        assert list(doc) == ["url", "title", "content", "raw_content", "metadata"]

    def test_page_error_shape(self):
        assert page_error("https://x.example", "boom") == {
            "url": "https://x.example",
            "title": "",
            "content": "",
            "error": "boom",
        }


# ---------------------------------------------------------------------------
# Setup schema
# ---------------------------------------------------------------------------


class TestSetupSchema:
    def test_with_env_var(self):
        schema = setup_schema("N", "paid", "tag", "KEY", "prompt", "https://k.example")
        assert schema == {
            "name": "N",
            "badge": "paid",
            "tag": "tag",
            "env_vars": [{"key": "KEY", "prompt": "prompt", "url": "https://k.example"}],
        }

    def test_without_env_var(self):
        assert setup_schema("N", "free", "tag")["env_vars"] == []


# ---------------------------------------------------------------------------
# Description formatting
# ---------------------------------------------------------------------------


class TestFormatDescription:
    def test_full_metadata(self):
        assert format_description(_FULL) == (
            "DeepSeek has released the V4 model.\n[Example News · 2026-06-01 · relevance 2.4/3 · authority 3/4]"
        )

    def test_falls_back_to_snippet(self):
        r = SearchResult(title="t", url="u", snippet="s", summary="", rerank_score=1.0)
        assert format_description(r).startswith("s\n[")

    def test_optional_metadata_omitted(self):
        r = SearchResult(title="t", url="u", summary="body")
        assert format_description(r) == "body\n[relevance 0.0/3]"

    def test_empty_body_only_metadata_line(self):
        r = SearchResult(title="t", url="u", rerank_score=1.5, authority=0)
        assert format_description(r) == "[relevance 1.5/3 · authority 0/4]"
