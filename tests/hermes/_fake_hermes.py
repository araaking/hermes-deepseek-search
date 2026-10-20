"""Minimal stand-in for Hermes' public ``agent.web_search_provider`` (ProviderBase + WebSearchProvider ABC +
get_provider_env), mirrored from hermes-agent@main so the adapter is testable without Hermes installed."""

from __future__ import annotations

import abc
import os
import sys
import types
from typing import Any


class ProviderBase(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def display_name(self) -> str:
        return self.name

    def get_setup_schema(self) -> dict[str, Any]:
        return {"name": self.display_name, "badge": "", "tag": "", "env_vars": []}


class WebSearchProvider(ProviderBase):
    @abc.abstractmethod
    def is_available(self) -> bool: ...

    def supports_search(self) -> bool:
        return True

    def is_keyless_available(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not support search (override supports_search)")

    def extract(self, urls: list[str], **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.name} does not support extract (override supports_extract)")


def get_provider_env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()


def install() -> None:
    """Register ``agent`` / ``agent.web_search_provider`` in ``sys.modules`` unless real Hermes is importable."""
    try:
        import agent.web_search_provider  # noqa: F401
    except ImportError:
        pass
    else:
        return
    pkg = types.ModuleType("agent")
    pkg.__path__ = []  # mark as package
    mod = types.ModuleType("agent.web_search_provider")
    mod.ProviderBase = ProviderBase
    mod.WebSearchProvider = WebSearchProvider
    mod.get_provider_env = get_provider_env
    pkg.web_search_provider = mod
    sys.modules.setdefault("agent", pkg)
    sys.modules.setdefault("agent.web_search_provider", mod)
