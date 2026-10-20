"""Tests for bench.run (MockTransport only — no network).

Covers:
- run_set: recorded fields, query_sent, dropped fields, per-question API and unexpected errors, partial output
- run_path label sanitising
- main: dry run, missing key, unreadable set, refuse to overwrite, argument checks, full run with summary
- the API key never reaches the output file
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from bench import run as bench_run
from bench.samples import SampleRow
from hermes_deepseek_search.core import DeepSeekClient

_ITEM = {
    "rerank_score": 2.4,
    "title": "Vikram Buddhi sentenced",
    "url": "https://example.com/buddhi",
    "snippet": "short",
    "summary": "He was sentenced in 2007.",
    "content": "",
    "site_name": "Example",
    "site_icon": "https://example.com/i.ico",
    "date": "2008-01-01T00:00:00+00:00",
    "authority": 3,
}
_ROWS = [
    SampleRow("dev-000", "Politics", "Date", "When was Vikram S. Buddhi sentenced?", "2007"),
    SampleRow("dev-001", "Art", "Person", "Who painted it?", "Someone"),
]


def _ok(data) -> httpx.Response:
    return httpx.Response(200, json=data)


class TestRunSet:
    async def test_records(self, make_client, recorder):
        records = await bench_run.run_set(_ROWS[:1], lambda: make_client(_ok({"results": [_ITEM, _ITEM]})), count=50)
        (record,) = records
        assert record["id"] == "dev-000"
        assert record["query_sent"] == "When was Vikram S. Buddhi sentenced?"
        assert record["query_truncated"] is False
        assert record["error"] is None
        assert isinstance(record["latency"], float)
        assert len(record["results"]) == 2
        assert set(record["results"][0]) == {
            "title",
            "url",
            "snippet",
            "summary",
            "rerank_score",
            "site_name",
            "date",
            "authority",
        }
        assert recorder.body()["count"] == 50

    async def test_api_error_recorded(self, make_client):
        error = httpx.Response(401, json={"error": {"message": "bad key", "type": "x", "param": None, "code": "x"}})
        records = await bench_run.run_set(_ROWS, lambda: make_client(error))
        assert [r["id"] for r in records] == ["dev-000", "dev-001"]
        assert all("HTTP 401" in r["error"] and r["results"] == [] for r in records)

    async def test_unexpected_error_recorded(self):
        def broken() -> DeepSeekClient:
            raise RuntimeError("factory boom")

        (record,) = await bench_run.run_set(_ROWS[:1], broken)
        assert record["error"] == "factory boom"

    async def test_fills_out_list(self, make_client):
        out = [None, None]
        await bench_run.run_set(_ROWS, lambda: make_client(_ok({"results": []})), out=out)
        assert [r["id"] for r in out] == ["dev-000", "dev-001"]


def test_run_path(tmp_path):
    now = datetime(2026, 10, 21, tzinfo=UTC)
    assert bench_run.run_path(tmp_path, "dev", "rerank v1/try", now).name == "2026-10-21-dev-rerank-v1-try.json"
    assert bench_run.run_path(tmp_path, "dev", "///", now).name == "2026-10-21-dev-run.json"


class TestMain:
    def test_dry_run(self, capsys, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert bench_run.main(["--set", "test", "--dry-run"]) == 0
        assert "test: 300 requests, at most $0.2250 (count=50)" in capsys.readouterr().out

    def test_missing_key(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert bench_run.main(["--set", "dev", "--limit", "1", "--out", str(tmp_path)]) == 2
        assert "uv run --env-file" in capsys.readouterr().err
        assert list(tmp_path.iterdir()) == []

    def test_unreadable_set(self, tmp_path, capsys):
        assert bench_run.main(["--set", str(tmp_path / "missing.csv")]) == 2
        assert "Cannot read sample set" in capsys.readouterr().err

    def test_argument_checks(self):
        with pytest.raises(SystemExit):
            bench_run.main(["--count", "51", "--dry-run"])

    def test_full_run(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-for-test")
        mock = httpx.MockTransport(lambda request: _ok({"results": [_ITEM]}))
        monkeypatch.setattr(bench_run, "DeepSeekClient", lambda key: DeepSeekClient(key, transport=mock))
        assert bench_run.main(["--set", "dev", "--limit", "2", "--label", "base", "--out", str(tmp_path)]) == 0
        (path,) = tmp_path.iterdir()
        text = path.read_text(encoding="utf-8")
        assert "sk-secret-for-test" not in text
        run = json.loads(text)
        assert run["meta"]["set"] == "dev"
        assert run["meta"]["label"] == "base"
        assert run["meta"]["complete"] is True
        assert run["meta"]["n"] == 2
        out = capsys.readouterr().out
        assert f"wrote {path}" in out
        assert "@1" in out
        # a second run with the same label refuses to overwrite
        assert bench_run.main(["--set", "dev", "--limit", "2", "--label", "base", "--out", str(tmp_path)]) == 2
        assert "already exists" in capsys.readouterr().err

    def test_interrupted_run_keeps_partial_results(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        def interrupted(coro):
            coro.close()
            raise KeyboardInterrupt

        monkeypatch.setattr(bench_run.asyncio, "run", interrupted)
        with pytest.raises(KeyboardInterrupt):
            bench_run.main(["--set", "dev", "--limit", "2", "--out", str(tmp_path)])
        (path,) = tmp_path.iterdir()
        run = json.loads(path.read_text(encoding="utf-8"))
        assert run["meta"]["complete"] is False
        assert run["rows"] == []
