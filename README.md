# hermes-deepseek-search

[![CI](https://github.com/araaking/hermes-deepseek-search/actions/workflows/ci.yml/badge.svg)](https://github.com/araaking/hermes-deepseek-search/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue.svg)

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that makes the
[DeepSeek Search & Fetch API](https://api-docs.deepseek.com/guides/search/) the backend for Hermes' built-in
`web_search` and `web_extract` tools. It uses the same `DEEPSEEK_API_KEY` as DeepSeek's models — no extra
account or key.

## Why

- **One key for models and search.** If Hermes already talks to DeepSeek models, web search needs nothing new.
- **Pay per successful request.** DeepSeek bills Search and Fetch per successful request ($0.75 per 1,000 at the
  time of writing, October 2026). Failed requests, empty searches and timeouts are not billed.
- **Model-oriented results.** Each hit carries a 600–800 character summary written for an LLM, plus relevance,
  authority and publication date.
- **Query operators** pass straight through: `site:`, `-site:`, `-term`, `"exact phrase"`, `-"phrase"`,
  `lang:zh,en`, `after:2025-01-01`, `before:2026`.

## Features

- One provider, `deepseek`, serving both `web_search` (`POST /v1/tools/search`) and `web_extract`
  (`POST /v1/tools/fetch`).
- Search descriptions = DeepSeek's summary plus a metadata line:
  `[site · YYYY-MM-DD · relevance x.y/3 · authority n/4]`.
- Parallel extract (5 at a time by default) under a 100 s batch deadline; one failing URL never fails the batch,
  and in-flight fetches are cancelled if Hermes times the call out.
- Each `web_search` call finishes within 60 s, retries included, so it never blocks Hermes for long.
- Retries with backoff for 408 / 429 / 5xx / network errors; clear, model-readable errors for 401 / 402 / 403 / 404.
- Local validation before spending a request: queries over 128 characters are shortened at a word boundary (the
  response carries a `notice`), invalid URLs are rejected without a call.
- Page text from Fetch is cleaned of Markdown backslash escapes (`\#`, `\-`, …) outside code blocks.
- Imports only Hermes' public plugin surface (`agent.web_search_provider`).
- Built-in end-to-end check: `hermes web-deepseek smoke`.
- Also an MCP server (`deepseek-search-mcp`) for Claude Code, Cursor, Claude Desktop and other MCP clients.

## Requirements

- Hermes Agent 0.22 or newer (also works on 0.21; see [Hermes 0.21](#hermes-021))
- A DeepSeek API key ([platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys))

## Installation

```bash
hermes plugins install araaking/hermes-deepseek-search --enable
hermes config set web.backend deepseek        # or web.search_backend / web.extract_backend separately
```

Hermes clones the repository to `~/.hermes/plugins/web-deepseek` and installs its Python dependency (`httpx`)
into the environment it manages, after asking you. If `DEEPSEEK_API_KEY` is not set yet, the installer asks for it
and saves it to `~/.hermes/.env`. Restart running Hermes sessions or the gateway afterwards.

Update with `hermes plugins update web-deepseek`; remove with `hermes plugins remove web-deepseek`.

### Hermes 0.21

Hermes 0.21 does not install a plugin's Python dependencies, so install the package into Hermes' virtualenv
instead, then enable it:

```bash
HPY=~/.hermes/hermes-agent/venv/bin/python     # Windows: %LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe
uv pip install --python "$HPY" "git+https://github.com/araaking/hermes-deepseek-search"
hermes plugins enable web-deepseek
hermes config set web.backend deepseek
```

Do not use this on 0.22: there, Hermes rebuilds its environment on updates and drops packages installed by hand.

## Verify

```bash
hermes web-deepseek smoke                  # isolated: throwaway HERMES_HOME, your config untouched
hermes web-deepseek smoke --current-home   # checks your real configuration
```

```
Hermes Agent 0.22.2 · hermes-deepseek-search 0.2.0 · isolated home (keyless rescue off)
[PASS] plugin registered — deepseek via plugin discovery
[PASS] missing-key error path (search) — DEEPSEEK_API_KEY is not set
[PASS] missing-key error path (extract) — DEEPSEEK_API_KEY is not set
[PASS] invalid-key error path — DeepSeek Search returned HTTP 401: ...
[PASS] API key available — present (value not shown)
[PASS] live web_search — 5 results; top: '...'
[PASS] live web_extract — 1,256 chars
[PASS] long-query notice — Query was longer than 128 characters and was shortened to: '...'
8 passed, 0 failed, 0 skipped
```

Exit code `0` means all passed, `1` a check failed, `2` Hermes is not importable (wrong interpreter). Live checks
cost about 3 successful requests (~$0.00225); `--skip-live` runs only the free checks. On Hermes 0.21 the same
check also runs as `"$HPY" -m hermes_deepseek_search.smoke`. The full walkthrough, including uninstalling, is in
[docs/server-verification.md](docs/server-verification.md).

## Configuration

Optional settings in `~/.hermes/config.yaml`:

```yaml
plugins:
  entries:
    web-deepseek:
      settings:
        min_rerank_score: 1.5
        extract_concurrency: 5
```

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `min_rerank_score` | `0.0` | 0–3 | Drop search hits below this DeepSeek score (1.5+ is high quality) |
| `max_retries` | `2` | 0–5 | Retries for 408 / 429 / 5xx / network errors |
| `base_url` | `https://api.deepseek.com` | `https://` URL (`http://` only for localhost), no credentials | API endpoint, e.g. for a proxy. It receives your API key; a non-DeepSeek host logs a warning |
| `extract_concurrency` | `5` | 1–25 | Parallel Fetch calls per `web_extract` |

Invalid values fall back to the default with a warning in the Hermes log.

## What the agent sees

A `web_search` result row:

```json
{
  "title": "DeepSeek V4 model released",
  "url": "https://example.com/deepseek-v4",
  "description": "DeepSeek has released the V4 model.\n[Example News · 2026-06-01 · relevance 2.4/3 · authority 3/4]",
  "position": 1
}
```

A `web_extract` entry is `{"url", "title", "content", "raw_content", "metadata"}` with the page as Markdown, or
`{"url", "title": "", "content": "", "error"}` when that URL failed.

## Keyless rescue (important)

When a configured web backend fails, Hermes may silently answer from a free keyless tier (for example Exa).
`web_search` still returns `"success": true` but adds `served_by`, `rescued_from` and `backend_error` under `data`;
`web_extract` gives no marker at all. Results that really came from DeepSeek always end each description with the
`[… relevance x.y/3 …]` line. To surface DeepSeek failures instead of masking them:

```bash
hermes config set web.keyless_fallback false
```

## MCP server (Claude Code, Cursor, Claude Desktop)

Outside Hermes, the same client runs as a stdio MCP server with two tools. It needs the `mcp` extra and
[uv](https://docs.astral.sh/uv/); `uvx` fetches and caches it on first start.

Claude Code:

```bash
claude mcp add deepseek-search -e DEEPSEEK_API_KEY=sk-... --   uvx --from "hermes-deepseek-search[mcp] @ git+https://github.com/araaking/hermes-deepseek-search" deepseek-search-mcp
```

Add `-s user` after `add` to make it available in every project, not just the current one.

Cursor (`~/.cursor/mcp.json`), Claude Desktop (`claude_desktop_config.json`) and most other clients:

```json
{
  "mcpServers": {
    "deepseek-search": {
      "command": "uvx",
      "args": [
        "--from",
        "hermes-deepseek-search[mcp] @ git+https://github.com/araaking/hermes-deepseek-search",
        "deepseek-search-mcp"
      ],
      "env": { "DEEPSEEK_API_KEY": "sk-..." }
    }
  }
}
```

| Tool | Parameters | Returns |
|---|---|---|
| `deepseek_search` | `query` (operators allowed), `count` 1–50 (default 10), `ori_question` (optional) | Numbered list: title, URL, summary and the `[site · date · relevance · authority]` line |
| `deepseek_fetch` | `url`, `max_chars` 1,000–500,000 (default 50,000) | `# Title`, `Source: <url>` and the page as Markdown, cut at `max_chars` with a marker |

Both tools are read-only, bill exactly like the Hermes plugin (per successful request), finish within 60 s
(search) or 100 s (fetch), and report DeepSeek errors such as `HTTP 401` as tool errors. Hermes users do not need
the MCP server: Hermes' own `web_search` / `web_extract` already go through the plugin.

## Limits

- Search: 100 QPS per account. Fetch: 25 QPS per account, plus a per-domain limit.
- Queries: 1–128 characters; at most 50 results per search.
- Extract returns Markdown only (Hermes' `format` argument is ignored) and fetches one page per URL — no crawling.

## Not supported (yet)

- A listing in the Hermes plugin catalog, or a PyPI package.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `hermes: invalid choice: 'web-deepseek'` | The plugin is not enabled: `hermes plugins enable web-deepseek` |
| `Hermes Agent is not importable` | Run the smoke test through Hermes: `hermes web-deepseek smoke` |
| `plugin registered` FAIL | Check `hermes plugins list`; `web-deepseek` must be installed and enabled |
| `HTTP 401` / `HTTP 402` | Invalid key / insufficient DeepSeek balance |
| `served by 'exa' (rescued from 'deepseek')` | DeepSeek failed; read `backend_error` (see Keyless rescue) |

More in [docs/server-verification.md](docs/server-verification.md#7-troubleshooting).

## Development

```bash
uv sync
uv run pytest                                  # unit tests: HTTP mocked, Hermes stubbed
uv run ruff check . && uv run ruff format --check .
DEEPSEEK_API_KEY=sk-... uv run pytest -m live  # real API, ~$0.0015
```

To try a working copy against a local Hermes without touching your setup, point `HERMES_HOME` at a scratch
directory that holds the checkout as a directory plugin:

```bash
mkdir -p /tmp/hds-home/plugins && cp -r . /tmp/hds-home/plugins/web-deepseek
printf 'plugins:\n  enabled:\n    - web-deepseek\n' > /tmp/hds-home/config.yaml
HERMES_HOME=/tmp/hds-home hermes web-deepseek smoke --skip-live
```

### Project layout

```
plugin.yaml    # Hermes directory-plugin manifest (id web-deepseek)
__init__.py    # directory-plugin wrapper: forwards register(ctx) to the package
src/hermes_deepseek_search/
├── core/      # Hermes-free async DeepSeek client: validation, errors, retry, models, Markdown cleanup
├── hermes/    # Hermes adapter: provider, settings, result envelopes, CLI command, register() entry point
├── mcp_server.py  # stdio MCP server (deepseek_search / deepseek_fetch), needs the [mcp] extra
└── smoke.py   # end-to-end check against a real Hermes install
tests/         # core, adapter (with a stubbed Hermes ABC) and smoke evaluators
```

## Disclaimer

Not affiliated with DeepSeek or Nous Research.

## License

[MIT](LICENSE)
