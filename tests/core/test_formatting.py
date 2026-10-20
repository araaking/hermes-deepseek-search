"""Tests for core.formatting (shared by the Hermes provider and the MCP server)."""

from __future__ import annotations

from hermes_deepseek_search.core import SearchResult, format_description, truncation_notice


def test_truncation_notice():
    assert truncation_notice("a b") == "Query was longer than 128 characters and was shortened to: 'a b'"


def test_format_description_full():
    result = SearchResult.from_api(
        {
            "title": "T",
            "url": "https://t.x",
            "summary": " Long summary. ",
            "site_name": "Site",
            "date": "2026-06-01T00:00:00+00:00",
            "rerank_score": 2.44,
            "authority": 3,
        }
    )
    assert format_description(result) == "Long summary.\n[Site · 2026-06-01 · relevance 2.4/3 · authority 3/4]"


def test_format_description_without_body_or_authority():
    result = SearchResult.from_api({"title": "T", "url": "https://t.x", "rerank_score": 0.5, "authority": -1})
    assert format_description(result) == "[relevance 0.5/3]"
