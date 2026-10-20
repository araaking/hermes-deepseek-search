# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-10-20

### Added
- Hermes 0.22 install route: `hermes plugins install araaking/hermes-deepseek-search`. The repository is now also
  a Hermes directory plugin (`plugin.yaml` plus a root `__init__.py` wrapper); Hermes installs `httpx` from
  `pyproject.toml` into its managed environment and prompts for `DEEPSEEK_API_KEY` when it is missing.
- `hermes web-deepseek smoke [flags]` runs the smoke test with Hermes' own interpreter, since Hermes 0.22 has no
  fixed venv path.
- `config_schema` in `plugin.yaml`, so Hermes Desktop shows the plugin settings as a form.

### Changed
- `web_extract` content no longer carries Markdown backslash escapes such as `\#` and `\-`; fenced code blocks
  and inline code are left as they are.

### Fixed
- `smoke --current-home` reported "no 'deepseek' provider" because it looked before Hermes had discovered plugins.
- Isolated smoke runs now find a directory-plugin install without `--register`.

## [0.1.0] — 2026-10-20

### Added
- `deepseek` web provider for Hermes Agent serving `web_search` (DeepSeek Search API) and `web_extract`
  (DeepSeek Fetch API), registered via the `web-deepseek` entry point.
- Async core client with local validation, typed errors and retry/backoff for 408/429/5xx/network errors.
- Search descriptions combine DeepSeek's model-oriented summary with site, date, relevance and authority.
- Parallel extract with a configurable concurrency cap and a 100 s batch deadline; in-flight fetches are
  cancelled when Hermes times the call out.
- 60 s overall deadline per `web_search` call; `Retry-After` honoured for 408, 429 and 5xx.
- `base_url` must be `https://` (plain `http://` only for localhost) and may not embed credentials.
- Optional settings: `min_rerank_score`, `max_retries`, `base_url`, `extract_concurrency`.
- `python -m hermes_deepseek_search.smoke` end-to-end check against a real Hermes install.
- Server installation and verification runbook (`docs/server-verification.md`).
