"""Typed results for the DeepSeek Search & Fetch API (tolerant of missing/null fields)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _str(value: Any) -> str:
    return "" if value is None else str(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    summary: str = ""
    content: str = ""
    rerank_score: float = 0.0
    site_name: str | None = None
    site_icon: str | None = None
    date: str | None = None
    authority: int = -1

    @classmethod
    def from_api(cls, item: Mapping[str, Any]) -> SearchResult:
        try:
            score = float(item.get("rerank_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            authority = int(item.get("authority", -1))
        except (TypeError, ValueError):
            authority = -1
        return cls(
            title=_str(item.get("title")),
            url=_str(item.get("url")),
            snippet=_str(item.get("snippet")),
            summary=_str(item.get("summary")),
            content=_str(item.get("content")),
            rerank_score=score,
            site_name=_opt_str(item.get("site_name")),
            site_icon=_opt_str(item.get("site_icon")),
            date=_opt_str(item.get("date")),
            authority=authority,
        )


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str  # the query actually sent (after normalization/truncation)
    query_truncated: bool
    results: list[SearchResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    title: str
    content: str
