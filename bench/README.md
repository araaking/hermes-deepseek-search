# Accuracy benchmark

Measures how often DeepSeek Search puts the correct answer in its first results, so that every change to the plugin
can be shown to raise (or not lower) accuracy. It calls the live API through the plugin's own `core` client. It is
not run in CI, because it needs a real key and costs money (a few cents per run).

## Sample sets (`data/`)

All three are drawn from OpenAI's [SimpleQA](https://github.com/openai/simple-evals) (4,326 questions, MIT; see
[`data/LICENSE-simpleqa.txt`](data/LICENSE-simpleqa.txt)) by `make_samples.py`, deterministically and without overlap:

| Set | File | Questions | Drawn as | Use |
|---|---|---|---|---|
| `dev` | `simpleqa_dev200.csv` | 200 (25 over 128 chars) | seed 20261020, first 200 — the original benchmark sample | tuning |
| `test` | `simpleqa_test300.csv` | 300 (40 over 128 chars) | the other 4,126, seed 20261021 | confirmation only |
| `long` | `simpleqa_long100.csv` | 100 (all over 128 chars) | remaining questions over 128 chars, seed 20261022 | long-query target |

Rule: try ideas and pick weights on **dev**. Run **test** and **long** once to confirm a choice; numbers chosen by
looking at the test set are not evidence.

A local set (for example Indonesian brand queries) is a CSV with the same columns plus `expected_urls`
(`|`-separated). Rows with expected URLs are scored by URL: a hit is a result URL equal to, or below, an expected one.

## Commands

```bash
# What a run would send and cost (no key needed)
uv run python -m bench.run --set dev --dry-run

# Live run: the key is read from DEEPSEEK_API_KEY only
uv run --env-file ~/.hermes/.env python -m bench.run --set dev --label baseline
#   → bench/runs/<YYYY-MM-DD>-dev-baseline.json  (git-ignored)

# Score a run, or compare two runs question by question (B is the candidate)
uv run python -m bench.score bench/runs/2026-10-21-dev-baseline.json
uv run python -m bench.score bench/runs/2026-10-21-dev-baseline.json --compare bench/runs/2026-10-21-dev-condensed.json
uv run python -m bench.score <run.json> --top 10     # as a count=10 caller would have seen it

# Rerank / fetch-N variants from one run, offline and free
uv run python -m bench.variants bench/runs/2026-10-21-dev-baseline.json
```

Each question is searched once with `count=50`, the maximum; DeepSeek charges per request, not per result, so the
extra results are free and let `bench.variants` test "fetch 20/30/50 and rerank" without new calls. Every result
field is kept (`title`, `url`, `snippet`, `summary`, `rerank_score`, `site_name`, `date`, `authority`).

| Set | Requests | Cost (at most) |
|---|---|---|
| dev | 200 | $0.15 |
| test | 300 | $0.225 |
| long | 100 | $0.075 |

Failed and empty requests are not billed, so the real cost can be lower.

## How a question is scored

- **Correct at rank N** when the normalised answer appears in the title, URL or summary/snippet of result N
  (`score.matches`, the original rule): lower case, accents removed, `3,000` = `3000`; a multi-word answer of 2–6
  significant words counts when all of them appear in the same result.
- Pages of the dataset itself are removed before ranking: `huggingface.co`, `kaggle.com` and
  `github.com/openai/simple-evals`. In the original run, 16 DeepSeek results were such pages.
- Reported: @1, @3, @5 (what Hermes shows by default), @10, MRR, the share of questions over 128 characters found
  in the top 10, errors, empty results, latency p50/p90 and cost, each rate with a 95% bootstrap interval.
- `--compare` pairs the two runs by question and runs an exact McNemar test per cut-off.

## Pitfalls

- **API drift.** The same query can return different results on different days. Always run the baseline and the
  candidate on the same day, and compare them with `--compare`.
- **False positives.** Text matching sometimes counts a result that only mentions the answer by chance (about 1–2 in
  16 in a manual check). The error is similar for both sides of a comparison.
- **Sample size.** With 200 questions a rate is known to about ±6 points; prefer McNemar over eyeballing deltas.
- **Index limits.** No rerank can find a page DeepSeek has not indexed.

## Files

| File | Purpose |
|---|---|
| `samples.py` | sample CSV format (`SampleRow`, `load`, `write`) |
| `make_samples.py` | regenerate `data/` from the official CSV (`--source` path or URL) |
| `run.py` | live runner (count=50, all fields, per-question errors, partial results kept on interrupt) |
| `score.py` | matching rule, dataset exclusion, metrics, bootstrap, McNemar, CLI |
| `variants.py` | offline rerank / fetch-N grid (prototype for the plugin's rerank) |
