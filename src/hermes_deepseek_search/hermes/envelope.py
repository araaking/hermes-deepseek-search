"""Hermes web-provider result shapes, replicated from the (private) ``plugins.web._common``.

Key order is part of the contract — it reaches the model as JSON. Keep these in step with
``agent/web_search_provider.py``'s documented envelope.
"""

from __future__ import annotations

from typing import Any

from hermes_deepseek_search.core import SearchResult, format_description


def search_ok(web_results: list[dict[str, Any]], *, notice: str | None = None) -> dict[str, Any]:
    envelope: dict[str, Any] = {"success": True, "data": {"web": web_results}}
    if notice:
        envelope["notice"] = notice
    return envelope


def search_fail(error: str) -> dict[str, Any]:
    return {"success": False, "error": error}


def title_hit(title: str, url: str, description: str, position: int) -> dict[str, Any]:
    return {"title": title, "url": url, "description": description, "position": position}


def document(url: str, title: str, content: str) -> dict[str, Any]:
    """Successful extract entry; ``raw_content`` mirrors ``content`` for the legacy pipeline."""
    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": content,
        "metadata": {"sourceURL": url, "title": title},
    }


def page_error(url: str, error: str) -> dict[str, Any]:
    return {"url": url, "title": "", "content": "", "error": error}


def setup_schema(name: str, badge: str, tag: str, key_env: str = "", prompt: str = "", url: str = "") -> dict[str, Any]:
    """``hermes tools`` picker entry; ``env_vars`` is empty when ``key_env`` is blank."""
    env_vars = [{"key": key_env, "prompt": prompt, "url": url}] if key_env else []
    return {"name": name, "badge": badge, "tag": tag, "env_vars": env_vars}


def search_rows(results: list[SearchResult]) -> list[dict[str, Any]]:
    """1-based positions, mirroring ``plugins.web._common.titled_rows``."""
    return [title_hit(r.title, r.url, format_description(r), i + 1) for i, r in enumerate(results)]
