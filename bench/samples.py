"""Benchmark sample sets: one CSV row per question.

Columns: ``id, topic, answer_type, question, answer`` and an optional ``expected_urls`` (``|``-separated). Rows with
expected URLs are scored by URL (local sets); the rest by answer text (SimpleQA).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SETS = {"dev": "simpleqa_dev200.csv", "test": "simpleqa_test300.csv", "long": "simpleqa_long100.csv"}
FIELDS = ("id", "topic", "answer_type", "question", "answer", "expected_urls")
_URL_SEP = "|"


@dataclass(frozen=True, slots=True)
class SampleRow:
    id: str
    topic: str
    answer_type: str
    question: str
    answer: str
    expected_urls: tuple[str, ...] = ()


def path_for(name_or_path: str) -> Path:
    """A set name from ``SETS`` or a path to any sample CSV (e.g. a local set added later)."""
    if name_or_path in SETS:
        return DATA_DIR / SETS[name_or_path]
    return Path(name_or_path)


def load(path: Path) -> list[SampleRow]:
    with open(path, encoding="utf-8", newline="") as fh:
        return [
            SampleRow(
                id=row["id"],
                topic=row.get("topic") or "",
                answer_type=row.get("answer_type") or "",
                question=row["question"],
                answer=row.get("answer") or "",
                expected_urls=tuple(u for u in (row.get("expected_urls") or "").split(_URL_SEP) if u.strip()),
            )
            for row in csv.DictReader(fh)
        ]


def write(path: Path, rows: list[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.id,
                    "topic": row.topic,
                    "answer_type": row.answer_type,
                    "question": row.question,
                    "answer": row.answer,
                    "expected_urls": _URL_SEP.join(row.expected_urls),
                }
            )
