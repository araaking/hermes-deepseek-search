"""Run a coroutine to completion from sync code (Hermes calls ``WebSearchProvider.search`` synchronously)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


def run_sync(factory: Callable[[], Awaitable[T]]) -> T:
    """``asyncio.run(factory())`` — or, when this thread already runs a loop (``asyncio.run`` would raise),
    the same on a short-lived worker thread. ``factory`` builds the coroutine inside the loop that awaits it,
    so no coroutine object or httpx client ever crosses loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(factory))
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-search") as pool:
        return pool.submit(lambda: asyncio.run(_await(factory))).result()


async def _await(factory: Callable[[], Awaitable[T]]) -> T:
    return await factory()
