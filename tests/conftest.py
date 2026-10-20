"""Shared fixtures: a DeepSeekClient wired to httpx.MockTransport with a recording no-op sleep."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from hermes_deepseek_search.core import DeepSeekClient


class Recorder:
    """Captures every request sent through the mock transport and every backoff sleep."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture
def make_client(recorder: Recorder) -> Callable[..., DeepSeekClient]:
    """``make_client(*responses, **client_kwargs)``: each item is an ``httpx.Response`` or an exception to
    raise, served in order; the last one repeats for any further request."""

    def factory(*responses: httpx.Response | Exception, **kwargs: Any) -> DeepSeekClient:
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.requests.append(request)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item

        async def fake_sleep(seconds: float) -> None:
            recorder.sleeps.append(seconds)

        return DeepSeekClient("sk-test", transport=httpx.MockTransport(handler), sleep=fake_sleep, **kwargs)

    return factory
