"""Model-oriented text shared by the adapters (Hermes provider, MCP server).

Lives in ``core`` because the Hermes adapter package cannot be imported outside Hermes.
"""

from __future__ import annotations

from hermes_deepseek_search.core.models import SearchResult

_DATE_CHARS = 10  # RFC 3339 "YYYY-MM-DD"


def format_description(result: SearchResult) -> str:
    """Model-oriented description: the long ``summary`` (falls back to ``snippet``) plus one metadata line
    ``[site · YYYY-MM-DD · relevance 2.4/3 · authority 3/4]`` so recency and trust survive Hermes' 4-field row."""
    body = (result.summary or result.snippet).strip()
    meta: list[str] = []
    if result.site_name:
        meta.append(result.site_name)
    if result.date:
        meta.append(result.date[:_DATE_CHARS])
    meta.append(f"relevance {result.rerank_score:.1f}/3")
    if result.authority >= 0:
        meta.append(f"authority {result.authority}/4")
    line = f"[{' · '.join(meta)}]"
    return f"{body}\n{line}" if body else line


def truncation_notice(query: str) -> str:
    """Told to the model when ``validate_query`` shortened its query, so it knows what was actually searched."""
    return f"Query was longer than 128 characters and was shortened to: {query!r}"
