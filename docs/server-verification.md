# Installing and verifying on a Hermes server

This runbook installs `hermes-deepseek-search` into an existing Hermes Agent install and proves that
`web_search` / `web_extract` are really served by DeepSeek. Commands assume Linux; on Windows the Hermes
home is `%LOCALAPPDATA%\hermes` and the interpreter is `hermes-agent\venv\Scripts\python.exe`.

## 1. Prerequisites

- Hermes Agent is installed (verified with 0.21.0). Find its interpreter:

  ```bash
  HPY=~/.hermes/hermes-agent/venv/bin/python
  $HPY -c "import hermes_cli; print(hermes_cli.__version__)"
  ```

- A DeepSeek API key, either exported as `DEEPSEEK_API_KEY` or stored in `~/.hermes/.env`
  (`DEEPSEEK_API_KEY=sk-...`). It is the same key Hermes uses for DeepSeek models.
- `uv` (recommended) or the venv's own `pip`.

## 2. Install into Hermes' venv

Pick one source:

```bash
# a) from a wheel built on your dev machine (`uv build` → dist/*.whl), copied to the server
uv pip install --python "$HPY" ./hermes_deepseek_search-0.1.0-py3-none-any.whl

# b) from GitHub
uv pip install --python "$HPY" "git+https://github.com/araaking/hermes-deepseek-search"

# c) without uv
"$HPY" -m pip install <wheel-or-git-url>
```

## 3. Verify in isolation (your config is not touched)

```bash
"$HPY" -m hermes_deepseek_search.smoke
```

The smoke test runs Hermes' real `web_search` / `web_extract` tools in a throwaway `HERMES_HOME` that selects
`deepseek`, enables the plugin and turns keyless rescue off. It reads your key the way Hermes does (environment,
then `~/.hermes/.env`) and never prints it. Live checks cost about 3 successful requests (~$0.00225); failed
requests are not billed.

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

Exit codes: `0` all passed, `1` a check failed, `2` Hermes is not importable (wrong interpreter).
Useful flags: `--skip-live` (free checks only), `--quick` (skip the notice check), `--query`, `--url`,
`--register` (bypass entry-point discovery to isolate a packaging problem), `--keep-home`.

## 4. Enable it for real

```bash
hermes plugins enable web-deepseek
hermes config set web.backend deepseek        # or set web.search_backend / web.extract_backend separately
```

If `plugins enable` refuses, add the id by hand in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - web-deepseek
```

Optional settings live under `plugins.entries.web-deepseek.settings`:

| Key | Default | Meaning |
|---|---|---|
| `min_rerank_score` | `0.0` | Drop search hits below this DeepSeek score (0–3; 1.5+ is high quality) |
| `max_retries` | `2` | Retries for 408 / 429 / 5xx / network errors (0–5) |
| `base_url` | `https://api.deepseek.com` | API endpoint (for proxies) |
| `extract_concurrency` | `5` | Parallel Fetch calls per `web_extract` (1–25) |

## 5. Verify the real configuration

```bash
"$HPY" -m hermes_deepseek_search.smoke --current-home
```

Then restart running Hermes sessions or the gateway and ask the agent about something from this week.

## 6. Keyless rescue — know what served the answer

When a configured web backend fails, Hermes may silently answer from a free keyless tier (for example Exa).
`web_search` still returns `"success": true` but adds `"served_by"` / `"rescued_from"` / `"backend_error"`
under `data`; `web_extract` gives no marker at all. Results that really came from DeepSeek always end each
description with a line like `[Example News · 2026-06-01 · relevance 2.4/3 · authority 3/4]`.

To surface DeepSeek failures instead of masking them:

```bash
hermes config set web.keyless_fallback false
```

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Hermes Agent is not importable` (exit 2) | Ran with the system Python | Use `$HPY` from step 1 |
| `plugin registered` FAIL | Package not in Hermes' venv, or `web-deepseek` not in `plugins.enabled` | Repeat step 2 with `--python "$HPY"`; check `hermes plugins list`. If `--register` passes, it is a packaging/discovery problem |
| `API key available` SKIP | Key not in the environment or `~/.hermes/.env` | `export DEEPSEEK_API_KEY=sk-...` or add it to `.env` |
| `HTTP 401` | Invalid key | Create a new key at https://platform.deepseek.com/api_keys |
| `HTTP 402` | Insufficient balance | Top up the DeepSeek account |
| `HTTP 429` | Search 100 QPS / Fetch 25 QPS account limit, or Fetch per-domain limit | Lower `extract_concurrency`; retries back off automatically |
| `served by 'exa' (rescued from 'deepseek')` | DeepSeek failed and Hermes rescued the call | Read `backend_error`; see section 6 |

## 8. Uninstall

```bash
hermes plugins disable web-deepseek
hermes config set web.backend <previous-backend>
uv pip uninstall --python "$HPY" hermes-deepseek-search
```
