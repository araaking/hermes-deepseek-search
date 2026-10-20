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
- Imports only Hermes' public plugin surface (`agent.web_search_provider`).
- Built-in end-to-end check: `python -m hermes_deepseek_search.smoke`.

## Requirements

- Hermes Agent (verified with 0.21.0)
- Python 3.11+ — i.e. Hermes' own virtualenv
- A DeepSeek API key ([platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys))

## Installation

Install into **Hermes' virtualenv**, not your system Python:

```bash
HPY=~/.hermes/hermes-agent/venv/bin/python
uv pip install --python "$HPY" "git+https://github.com/araaking/hermes-deepseek-search"
# without uv:
"$HPY" -m pip install "git+https://github.com/araaking/hermes-deepseek-search"
```

On Windows the interpreter is `%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\python.exe`.

## Enable

```bash
hermes plugins enable web-deepseek
hermes config set web.backend deepseek        # or web.search_backend / web.extract_backend separately
```

Make sure `DEEPSEEK_API_KEY` is set in the environment or in `~/.hermes/.env`, then restart running Hermes
sessions or the gateway.

## Verify

```bash
"$HPY" -m hermes_deepseek_search.smoke                 # isolated: throwaway HERMES_HOME, your config untouched
"$HPY" -m hermes_deepseek_search.smoke --current-home  # after enabling: checks your real configuration
```

```
Hermes Agent 0.21.0 · hermes-deepseek-search 0.1.0 · isolated home (keyless rescue off)
[PASS] plugin registered — deepseek via entry point
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
cost about 3 successful requests (~$0.00225); `--skip-live` runs only the free checks. The full walkthrough —
including uninstalling — is in [docs/server-verification.md](docs/server-verification.md).

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

## Limits

- Search: 100 QPS per account. Fetch: 25 QPS per account, plus a per-domain limit.
- Queries: 1–128 characters; at most 50 results per search.
- Extract returns Markdown only (Hermes' `format` argument is ignored) and fetches one page per URL — no crawling.

## Not supported (yet)

- `hermes plugins install <git-url>`: Hermes 0.21.0 does not install a git plugin's Python dependencies, so install
  with pip/uv as shown above.
- A PyPI package.
- An MCP server for non-Hermes clients.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Hermes Agent is not importable` | Run the smoke test with Hermes' interpreter (`$HPY`) |
| `plugin registered` FAIL | Install into Hermes' venv and add `web-deepseek` to `plugins.enabled` |
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

To try the package against a local Hermes without installing it, build a wheel, unzip it into a scratch
directory, and run the smoke test with Hermes' interpreter and that directory on `PYTHONPATH`:

```bash
uv build
python -c "import zipfile,glob; zipfile.ZipFile(glob.glob('dist/*.whl')[0]).extractall('/tmp/hds-site')"
PYTHONPATH=/tmp/hds-site "$HPY" -m hermes_deepseek_search.smoke --skip-live
```

### Project layout

```
src/hermes_deepseek_search/
├── core/      # Hermes-free async DeepSeek client: validation, errors, retry, models
├── hermes/    # Hermes adapter: provider, settings, result envelopes, register() entry point
└── smoke.py   # end-to-end check against a real Hermes install
tests/         # core, adapter (with a stubbed Hermes ABC) and smoke evaluators
```

## Disclaimer

Not affiliated with DeepSeek or Nous Research.

## License

[MIT](LICENSE)
