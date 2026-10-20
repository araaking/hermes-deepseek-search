"""Draw the committed SimpleQA sample sets from OpenAI's official CSV (deterministic; already done, see bench/data).

- dev 200: ``random.Random(20261020).shuffle(rows)``, first 200 — the owner's original benchmark sample.
- test 300: the remaining 4,126 shuffled with seed 20261021, first 300.
- long 100: questions over 128 characters from what is left after test, shuffled with seed 20261022, first 100.

Run: ``uv run python -m bench.make_samples [--source <csv path or URL>] [--out bench/data]``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import random
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from bench.samples import DATA_DIR, SETS, SampleRow, write

SOURCE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
DEV_SEED, TEST_SEED, LONG_SEED = 20261020, 20261021, 20261022
DEV_SIZE, TEST_SIZE, LONG_SIZE = 200, 300, 100
LONG_CHARS = 128  # DeepSeek's query limit; longer questions used to be cut


def read_source(source: str) -> list[dict[str, str]]:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=60) as resp:
            text = resp.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _row(raw: dict[str, str], ident: str) -> SampleRow:
    meta = ast.literal_eval(raw["metadata"])
    return SampleRow(
        id=ident,
        topic=str(meta.get("topic", "")),
        answer_type=str(meta.get("answer_type", "")),
        question=raw["problem"],
        answer=raw["answer"],
    )


def draw(
    raw_rows: list[dict[str, str]],
    *,
    dev_size: int = DEV_SIZE,
    test_size: int = TEST_SIZE,
    long_size: int = LONG_SIZE,
) -> dict[str, list[SampleRow]]:
    rows = list(raw_rows)
    random.Random(DEV_SEED).shuffle(rows)
    dev, rest = rows[:dev_size], rows[dev_size:]
    random.Random(TEST_SEED).shuffle(rest)
    test = rest[:test_size]
    long_pool = [r for r in rest[test_size:] if len(r["problem"]) > LONG_CHARS]
    random.Random(LONG_SEED).shuffle(long_pool)
    sets = {
        "dev": [_row(r, f"dev-{i:03d}") for i, r in enumerate(dev)],
        "test": [_row(r, f"test-{i:03d}") for i, r in enumerate(test)],
        "long": [_row(r, f"long-{i:03d}") for i, r in enumerate(long_pool[:long_size])],
    }
    questions = [row.question for rows_ in sets.values() for row in rows_]
    if len(questions) != len(set(questions)):
        raise ValueError("sample sets overlap")
    return sets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.make_samples", description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=SOURCE_URL, help="official SimpleQA CSV (path or URL)")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help="output directory")
    args = parser.parse_args(argv)
    sets = draw(read_source(args.source))
    for name, rows in sets.items():
        write(args.out / SETS[name], rows)
        long_n = sum(1 for r in rows if len(r.question) > LONG_CHARS)
        print(f"{name:5} {len(rows):4} questions, {long_n} over {LONG_CHARS} chars -> {args.out / SETS[name]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
