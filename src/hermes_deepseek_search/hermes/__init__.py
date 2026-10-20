"""Hermes Agent plugin entry point: registers DeepSeek as the ``deepseek`` web search/extract backend.

Enable with ``hermes plugins enable web-deepseek``, then ``hermes config set web.backend deepseek``.
"""

from __future__ import annotations

from typing import Any

from hermes_deepseek_search.hermes.provider import DeepSeekWebSearchProvider
from hermes_deepseek_search.hermes.settings import ProviderSettings

__all__ = ["DeepSeekWebSearchProvider", "register"]


def register(ctx: Any) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_web_search_provider(DeepSeekWebSearchProvider(ProviderSettings.from_ctx(ctx)))
