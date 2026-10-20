"""Tests for bench.samples and bench.make_samples.

Covers:
- CSV round trip including optional expected_urls; set names vs paths
- draw(): deterministic, disjoint sets, long set only long questions, duplicate questions rejected
- the committed data files (sizes, disjointness, long set)
- make_samples CLI with a local source and with a URL source
"""

from __future__ import annotations

import csv
import io

import pytest

from bench import make_samples, samples
from bench.samples import SampleRow


def _raw(i: int, *, long: bool = False) -> dict[str, str]:
    question = f"Question {i} " + ("about a very long topic " * 8 if long else "short?")
    return {"metadata": "{'topic': 'Art', 'answer_type': 'Date'}", "problem": question, "answer": f"answer {i}"}


def _toy(n: int = 40) -> list[dict[str, str]]:
    return [_raw(i, long=i % 3 == 0) for i in range(n)]


class TestSampleFiles:
    def test_round_trip(self, tmp_path):
        rows = [
            SampleRow("a", "Art", "Date", "Q1?", "A1"),
            SampleRow("b", "", "", "Q2, with comma", "A2", ("https://x.example", "https://y.example/p")),
        ]
        path = tmp_path / "s.csv"
        samples.write(path, rows)
        assert samples.load(path) == rows

    def test_load_without_optional_columns(self, tmp_path):
        path = tmp_path / "local.csv"
        path.write_text("id,question,expected_urls\nl1,sozo jakarta,https://sozo.id\n", encoding="utf-8")
        assert samples.load(path) == [SampleRow("l1", "", "", "sozo jakarta", "", ("https://sozo.id",))]

    def test_path_for(self, tmp_path):
        assert samples.path_for("dev") == samples.DATA_DIR / "simpleqa_dev200.csv"
        assert samples.path_for(str(tmp_path / "x.csv")) == tmp_path / "x.csv"


class TestDraw:
    def test_deterministic_and_disjoint(self):
        first = make_samples.draw(_toy(), dev_size=5, test_size=6, long_size=4)
        assert first == make_samples.draw(_toy(), dev_size=5, test_size=6, long_size=4)
        sizes = {name: len(rows) for name, rows in first.items()}
        assert sizes == {"dev": 5, "test": 6, "long": 4}
        questions = [r.question for rows in first.values() for r in rows]
        assert len(questions) == len(set(questions))
        assert all(len(r.question) > make_samples.LONG_CHARS for r in first["long"])
        assert first["dev"][0].id == "dev-000"
        assert (first["dev"][0].topic, first["dev"][0].answer_type) == ("Art", "Date")

    def test_duplicate_questions_rejected(self):
        rows = [_raw(1)] * 4
        with pytest.raises(ValueError, match="overlap"):
            make_samples.draw(rows, dev_size=2, test_size=2, long_size=0)


class TestCommittedData:
    def test_sets(self):
        sets = {name: samples.load(samples.path_for(name)) for name in samples.SETS}
        assert {name: len(rows) for name, rows in sets.items()} == {"dev": 200, "test": 300, "long": 100}
        questions = [r.question for rows in sets.values() for r in rows]
        assert len(questions) == len(set(questions))
        assert all(len(r.question) > 128 for r in sets["long"])
        assert sum(len(r.question) > 128 for r in sets["dev"]) == 25
        assert sets["dev"][0].question == "What was the year when Vikram S. Buddhi was sentenced to prison?"


class TestCli:
    def test_local_source(self, tmp_path, capsys):
        source = tmp_path / "src.csv"
        with open(source, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["metadata", "problem", "answer"])
            writer.writeheader()
            writer.writerows(_toy(30))
        assert make_samples.main(["--source", str(source), "--out", str(tmp_path / "out")]) == 0
        assert "dev     30 questions" in capsys.readouterr().out
        assert len(samples.load(tmp_path / "out" / "simpleqa_dev200.csv")) == 30

    def test_url_source(self, monkeypatch):
        body = io.StringIO()
        writer = csv.DictWriter(body, fieldnames=["metadata", "problem", "answer"])
        writer.writeheader()
        writer.writerows(_toy(3))
        seen = {}

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(url, timeout):
            seen.update(url=url, timeout=timeout)
            return Response(body.getvalue().encode("utf-8"))

        monkeypatch.setattr(make_samples.urllib.request, "urlopen", fake_urlopen)
        rows = make_samples.read_source("https://example.com/simple_qa.csv")
        assert len(rows) == 3
        assert seen == {"url": "https://example.com/simple_qa.csv", "timeout": 60}
