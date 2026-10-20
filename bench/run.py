"""Run a benchmark set against the live DeepSeek Search API through the project's own core client.

Each question is searched once with ``count=50`` (the same price as ``count=10``) and every result field is kept, so
rerank and "fetch N" variants can be scored offline afterwards (``bench.variants``). The API key is read from the
``DEEPSEEK_API_KEY`` environment variable only, for example::

    uv run --env-file ~/.hermes/.env python -m bench.run --set dev --label baseline

Output: ``bench/runs/<YYYY-MM-DD>-<set>-<label>.json`` (git-ignored). At most $0.00075 per question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench import samples, score
from bench.samples import SampleRow
from hermes_deepseek_search import __version__
from hermes_deepseek_search.core import API_KEY_ENV, DeepSeekClient, SearchResult

RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_COUNT = 50  # DeepSeek's maximum; price does not depend on it
DEFAULT_CONCURRENCY = 8  # far below the 100 QPS Search limit
_DROPPED_FIELDS = ("content", "site_icon")  # empty without need_content / no ranking value


def _result(result: SearchResult) -> dict[str, Any]:
    data = asdict(result)
    for name in _DROPPED_FIELDS:
        data.pop(name, None)
    return data


async def run_set(
    rows: Sequence[SampleRow],
    client_factory: Callable[[], DeepSeekClient],
    *,
    count: int = DEFAULT_COUNT,
    concurrency: int = DEFAULT_CONCURRENCY,
    out: list[dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Search every row; ``out`` (pre-sized) receives records as they finish, so an interrupted run keeps them."""
    out = out if out is not None else [None] * len(rows)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, row: SampleRow) -> None:
        record: dict[str, Any] = {
            "id": row.id,
            "question": row.question,
            "answer": row.answer,
            "topic": row.topic,
            "answer_type": row.answer_type,
            "expected_urls": list(row.expected_urls),
            "query_sent": None,
            "query_truncated": False,
            "latency": None,
            "error": None,
            "results": [],
        }
        async with semaphore:
            started = time.perf_counter()
            try:
                async with client_factory() as client:
                    resp = await client.search(row.question, count=count)
            except Exception as exc:  # recorded per question; the run goes on
                record["error"] = str(exc) or type(exc).__name__
            else:
                record["query_sent"] = resp.query
                record["query_truncated"] = resp.query_truncated
                record["results"] = [_result(r) for r in resp.results]
            record["latency"] = round(time.perf_counter() - started, 3)
        out[index] = record

    await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
    return [r for r in out if r is not None]


def run_path(out_dir: Path, set_name: str, label: str, now: datetime) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "run"
    return out_dir / f"{now:%Y-%m-%d}-{set_name}-{safe_label}.json"


def write_run(path: Path, run: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    score.utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m bench.run", description=__doc__.splitlines()[0])
    parser.add_argument("--set", default="dev", help=f"one of {', '.join(samples.SETS)} or a sample CSV path")
    parser.add_argument("--label", default="baseline", help="name for this run (part of the file name)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="results per query (1-50)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite an existing run file")
    parser.add_argument("--dry-run", action="store_true", help="show what would be sent and the cost, then stop")
    args = parser.parse_args(argv)
    if not 1 <= args.count <= DEFAULT_COUNT or args.concurrency < 1:
        parser.error("--count must be 1-50 and --concurrency at least 1")

    try:
        rows = samples.load(samples.path_for(args.set))
    except (OSError, KeyError) as exc:
        print(f"Cannot read sample set {args.set!r}: {exc}", file=sys.stderr)
        return 2
    rows = rows[: args.limit] if args.limit else rows
    set_name = args.set if args.set in samples.SETS else samples.path_for(args.set).stem
    print(f"{set_name}: {len(rows)} requests, at most ${len(rows) * score.PRICE_PER_REQUEST:.4f} (count={args.count})")
    if args.dry_run:
        return 0

    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        print(f"Set {API_KEY_ENV}, e.g. uv run --env-file ~/.hermes/.env python -m bench.run ...", file=sys.stderr)
        return 2
    now = datetime.now(UTC)
    path = run_path(args.out, set_name, args.label, now)
    if path.exists() and not args.force:
        print(f"{path} already exists; pick another --label or pass --force", file=sys.stderr)
        return 2

    out: list[dict[str, Any] | None] = [None] * len(rows)
    meta = {
        "set": set_name,
        "label": args.label,
        "count": args.count,
        "created_utc": now.isoformat(timespec="seconds"),
        "package_version": __version__,
    }
    try:
        asyncio.run(
            run_set(rows, lambda: DeepSeekClient(api_key), count=args.count, concurrency=args.concurrency, out=out)
        )
    finally:
        done = [r for r in out if r is not None]
        run = {"meta": {**meta, "n": len(done), "complete": len(done) == len(rows)}, "rows": done}
        write_run(path, run)
    print(f"wrote {path}")
    print(score.summary(run))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
