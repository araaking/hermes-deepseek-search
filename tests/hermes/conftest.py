"""Fixtures for the Hermes adapter: a fake ``agent.web_search_provider`` (installed before any adapter import),
a provider wired to httpx.MockTransport, and a recording PluginContext stand-in."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.hermes import _fake_hermes

_fake_hermes.install()

from hermes_deepseek_search.core import DeepSeekClient  # noqa: E402
from hermes_deepseek_search.hermes.provider import DeepSeekWebSearchProvider  # noqa: E402
from hermes_deepseek_search.hermes.settings import ProviderSettings  # noqa: E402


class FakeContext:
    """Records ``register_web_search_provider`` calls; ``get_config`` serves a flat settings dict."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = settings or {}
        self.providers: list[Any] = []

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def register_web_search_provider(self, provider: Any) -> None:
        self.providers.append(provider)


@pytest.fixture
def api_key(monkeypatch) -> str:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    return "sk-test"


@pytest.fixture
def make_provider(recorder):
    """``make_provider(replies, settings=None, **provider_kwargs)``.

    ``replies``: a callable ``(httpx.Request) -> httpx.Response`` (for URL-dependent replies) or a list of
    responses/exceptions served in order (last repeats). The client never really sleeps between retries."""

    def factory(replies, settings: ProviderSettings | None = None, **kwargs: Any) -> DeepSeekWebSearchProvider:
        if callable(replies):

            def handler(request: httpx.Request) -> httpx.Response:
                recorder.requests.append(request)
                return replies(request)

        else:
            queue = list(replies)

            def handler(request: httpx.Request) -> httpx.Response:
                recorder.requests.append(request)
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, Exception):
                    raise item
                return item

        async def no_sleep(seconds: float) -> None:
            recorder.sleeps.append(seconds)

        s = settings or ProviderSettings()

        def client_factory(key: str) -> DeepSeekClient:
            return DeepSeekClient(
                key,
                base_url=s.base_url,
                max_retries=s.max_retries,
                transport=httpx.MockTransport(handler),
                sleep=no_sleep,
            )

        return DeepSeekWebSearchProvider(s, client_factory=client_factory, **kwargs)

    return factory
