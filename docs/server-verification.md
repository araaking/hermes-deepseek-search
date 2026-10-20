# Installing and verifying on a Hermes server

This runbook installs `hermes-deepseek-search` into an existing Hermes Agent install and proves that
`web_search` / `web_extract` are really served by DeepSeek. Commands assume Linux; on Windows the Hermes
home is `%LOCALAPPDATA%\hermes`.

## 1. Prerequisites

- Hermes Agent 0.22 or newer (`hermes --version`). For 0.21, see [section 9](#9-hermes-021).
- A DeepSeek API key, either exported as `DEEPSEEK_API_KEY` or stored in `~/.hermes/.env`
  (`DEEPSEEK_API_KEY=sk-...`). It is the same key Hermes uses for DeepSeek models. If it is missing, the
  installer in step 2 asks for it and saves it to `.env`.

## 2. Install

```bash
hermes plugins install araaking/hermes-deepseek-search --enable
```

Hermes clones the repository to `~/.hermes/plugins/web-deepseek`, scans it, and asks before installing the
Python dependency (`httpx`) into the environment it manages. For an unattended install, add `--yes-deps`.
To pin a reviewed commit, add `--ref <40-character SHA>`.

If you copied the plugin into `~/.hermes/plugins/web-deepseek` by hand before this route existed, remove that
copy first (`hermes plugins remove web-deepseek`, or delete the directory), then install as above so updates
work.

## 3. Verify in isolation (your config is not touched)

```bash
hermes web-deepseek smoke
```

The smoke test runs Hermes' real `web_search` / `web_extract` tools in a throwaway `HERMES_HOME` that selects
`deepseek`, enables the plugin and turns keyless rescue off. It reads your key the way Hermes does (environment,
then `~/.hermes/.env`) and never prints it. Live checks cost about 3 successful requests (~$0.00225); failed
requests are not billed.

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

Exit codes: `0` all passed, `1` a check failed, `2` Hermes is not importable. Useful flags: `--skip-live` (free
checks only), `--quick` (skip the notice check), `--query`, `--url`, `--register` (bypass plugin discovery to
isolate a packaging problem), `--keep-home`.

## 4. Select DeepSeek as the web backend

```bash
hermes config set web.backend deepseek        # or set web.search_backend / web.extract_backend separately
```

If you installed without `--enable`, also run `hermes plugins enable web-deepseek`.

Optional settings live under `plugins.entries.web-deepseek.settings` (Hermes Desktop shows them as a form):

| Key | Default | Meaning |
|---|---|---|
| `min_rerank_score` | `0.0` | Drop search hits below this DeepSeek score (0–3; 1.5+ is high quality) |
| `max_retries` | `2` | Retries for 408 / 429 / 5xx / network errors (0–5) |
| `base_url` | `https://api.deepseek.com` | API endpoint (for proxies). Must be `https://` (`http://` only for localhost) with no credentials in the URL — it receives your API key |
| `extract_concurrency` | `5` | Parallel Fetch calls per `web_extract` (1–25) |

## 5. Verify the real configuration

```bash
hermes web-deepseek smoke --current-home
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
| `hermes: invalid choice: 'web-deepseek'` | Plugin not installed or not enabled | `hermes plugins list`; `hermes plugins enable web-deepseek` |
| `plugin registered` FAIL | Plugin disabled, or its dependency install was declined | `hermes plugins list`; reinstall and accept the dependency prompt. If `--register` passes, it is a discovery problem |
| `Hermes Agent is not importable` (exit 2) | `python -m hermes_deepseek_search.smoke` run with the wrong Python | Use `hermes web-deepseek smoke` |
| `API key available` SKIP | Key not in the environment or `~/.hermes/.env` | `export DEEPSEEK_API_KEY=sk-...` or add it to `.env` |
| `HTTP 401` | Invalid key | Create a new key at https://platform.deepseek.com/api_keys |
| `HTTP 402` | Insufficient balance | Top up the DeepSeek account |
| `HTTP 429` | Search 100 QPS / Fetch 25 QPS account limit, or Fetch per-domain limit | Lower `extract_concurrency`; retries back off automatically |
| `served by 'exa' (rescued from 'deepseek')` | DeepSeek failed and Hermes rescued the call | Read `backend_error`; see section 6 |

## 8. Update and uninstall

```bash
hermes plugins update web-deepseek

hermes config set web.backend <previous-backend>
hermes plugins remove web-deepseek
```

## 9. Hermes 0.21

Hermes 0.21 does not install a plugin's Python dependencies, so the package goes into Hermes' virtualenv:

```bash
HPY=~/.hermes/hermes-agent/venv/bin/python    # Windows: hermes-agent\venv\Scripts\python.exe
uv pip install --python "$HPY" "git+https://github.com/araaking/hermes-deepseek-search"   # or: "$HPY" -m pip install ...
hermes plugins enable web-deepseek
hermes config set web.backend deepseek
hermes web-deepseek smoke --current-home      # or: "$HPY" -m hermes_deepseek_search.smoke --current-home
```

Uninstall with `uv pip uninstall --python "$HPY" hermes-deepseek-search` after disabling the plugin. Do not use
this route on 0.22: Hermes rebuilds its environment on updates and drops packages installed by hand.
