"""MCP server exposing DeepSeek Search & Fetch as ``deepseek_search`` and ``deepseek_fetch`` (stdio).

For MCP clients other than Hermes (Claude Code, Cursor, Claude Desktop …). Needs the ``mcp`` extra; the full
``uvx --from "hermes-deepseek-search[mcp] @ git+https://github.com/…" deepseek-search-mcp`` command is in the README.

``DEEPSEEK_API_KEY`` is read on every call, so a missing key is a tool error rather than a startup failure. stdout
carries the protocol; logs go to stderr. Postponed annotations stay off in this module: the MCP SDK reads the tool
parameter hints when the tools are registered, and ``Field`` is imported lazily inside ``build_server``.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Annotated

from hermes_deepseek_search import __version__
from hermes_deepseek_search.core import (
    DeepSeekClient,
    DeepSeekError,
    FetchResult,
    MissingApiKeyError,
    SearchResponse,
    format_description,
    truncation_notice,
    unescape_markdown,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

logger = logging.getLogger(__name__)

SERVER_NAME = "deepseek-search"
_MAX_SEARCH_COUNT = 50  # DeepSeek caps count at 50
_SEARCH_DEADLINE_S = 60.0  # one call never hangs the client longer than this, retries included
_FETCH_DEADLINE_S = 100.0  # Fetch is slower (target site + Markdown conversion)
_DEFAULT_MAX_CHARS = 50_000  # keeps one page from flooding the client's context
_MIN_MAX_CHARS = 1_000
_MAX_MAX_CHARS = 500_000
_INDENT = "   "
_EXTRA_HINT = (
    "The MCP server needs the 'mcp' extra: uvx --from "
    '"hermes-deepseek-search[mcp] @ git+https://github.com/araaking/hermes-deepseek-search" deepseek-search-mcp'
)
_KEY_HINT = "Set it in the MCP server's env, e.g. claude mcp add deepseek-search -e DEEPSEEK_API_KEY=sk-... -- ..."
_INSTRUCTIONS = (
    "Use deepseek_search for current or recent information from the web. Queries support the operators site:, "
    '-site:, "exact phrase", -term, lang:zh,en, after:2025-01-01 and before:2026. Use deepseek_fetch to read a '
    "result page in full."
)


def format_search(resp: SearchResponse) -> str:
    """Numbered Markdown list: title, URL, then the summary and ``[site · date · relevance · authority]`` line."""
    if not resp.results:
        lines = [f"No results for {resp.query!r}."]
    else:
        lines = [f"Results for {resp.query!r}: {len(resp.results)}"]
    if resp.query_truncated:
        lines.append(truncation_notice(resp.query))
    for i, result in enumerate(resp.results, start=1):
        description = format_description(result).replace("\n", "\n" + _INDENT)
        lines.append(f"\n{i}. {result.title}\n{_INDENT}{result.url}\n{_INDENT}{description}")
    return "\n".join(lines)


def format_page(page: FetchResult, max_chars: int) -> str:
    """``# Title``, the source URL and the page as cleaned Markdown, cut after ``max_chars`` characters."""
    content = unescape_markdown(page.content).strip()
    if not content:
        body = "(no extractable content)"
    elif len(content) > max_chars:
        body = (
            f"{content[:max_chars]}\n\n[Truncated: showing {max_chars:,} of {len(content):,} characters. "
            "Call again with a larger max_chars to read more.]"
        )
    else:
        body = content
    return f"# {page.title or page.url}\nSource: {page.url}\n\n{body}"


def build_server(
    client_factory: Callable[[], DeepSeekClient] = DeepSeekClient.from_env,
    *,
    search_deadline: float = _SEARCH_DEADLINE_S,
    fetch_deadline: float = _FETCH_DEADLINE_S,
) -> "MCPServer":
    """The server with both tools registered. Raises ``ModuleNotFoundError`` when the ``mcp`` extra is missing."""
    from mcp.server import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations
    from pydantic import Field

    server = MCPServer(SERVER_NAME, instructions=_INSTRUCTIONS, version=__version__)
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=True)

    def tool_error(exc: Exception, label: str, deadline: float) -> ToolError:
        if isinstance(exc, TimeoutError):
            return ToolError(f"DeepSeek {label} did not finish within {deadline:g}s")
        if isinstance(exc, MissingApiKeyError):
            return ToolError(f"{exc} {_KEY_HINT}")
        if isinstance(exc, DeepSeekError):
            return ToolError(str(exc))
        logger.warning("DeepSeek %s error: %s", label.lower(), exc)
        return ToolError(f"DeepSeek {label.lower()} failed: {exc}")

    @server.tool(title="DeepSeek web search", annotations=read_only, structured_output=False)
    async def deepseek_search(
        query: Annotated[
            str,
            Field(
                description="Search query, 1-128 characters (longer queries are shortened). Supports site:, "
                '-site:, "exact phrase", -term, lang:, after: and before:.'
            ),
        ],
        count: Annotated[
            int, Field(ge=1, le=_MAX_SEARCH_COUNT, description=f"Number of results, 1-{_MAX_SEARCH_COUNT}")
        ] = 10,
        ori_question: Annotated[
            str | None, Field(description="The user's original question, to improve result ranking")
        ] = None,
    ) -> str:
        """Search the live web with DeepSeek. Each result has a title, URL, a summary written for a model and a
        [site · date · relevance x/3 · authority n/4] line (relevance 1.5+ is high quality)."""
        try:
            async with asyncio.timeout(search_deadline):
                async with client_factory() as client:
                    resp = await client.search(query, count=count, ori_question=ori_question)
        except Exception as exc:
            raise tool_error(exc, "Search", search_deadline) from exc
        logger.info("DeepSeek Search: %d results (count %d)", len(resp.results), count)
        return format_search(resp)

    @server.tool(title="DeepSeek fetch page", annotations=read_only, structured_output=False)
    async def deepseek_fetch(
        url: Annotated[str, Field(description="http(s) URL of the page to read")],
        max_chars: Annotated[
            int,
            Field(
                ge=_MIN_MAX_CHARS,
                le=_MAX_MAX_CHARS,
                description="Cut the page text after this many characters",
            ),
        ] = _DEFAULT_MAX_CHARS,
    ) -> str:
        """Read one web page with DeepSeek and return it as Markdown."""
        try:
            async with asyncio.timeout(fetch_deadline):
                async with client_factory() as client:
                    page = await client.fetch(url)
        except Exception as exc:
            raise tool_error(exc, "Fetch", fetch_deadline) from exc
        logger.info("DeepSeek Fetch: %d chars", len(page.content))
        return format_page(page, max_chars)

    return server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepseek-search-mcp", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)
    try:
        server = build_server()
    except ModuleNotFoundError as exc:
        if exc.name != "mcp" and not (exc.name or "").startswith("mcp."):
            raise
        print(_EXTRA_HINT, file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
