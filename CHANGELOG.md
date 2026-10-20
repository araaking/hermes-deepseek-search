# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

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
