"""Tests for bench.score.

Covers:
- The owner's matching rule (normalisation, multi-word answers, stop words, the 6-word limit)
- Dataset-page exclusion and its effect on ranks; URL scoring for local sets
- score_run / metrics (top-N cut, errors, empty results, cost, latency, long-query split)
- bootstrap and exact McNemar (values from the original analysis)
- summary / compare reports and the CLI
"""

from __future__ import annotations

import json

import pytest

from bench import score


def _r(title="", url="https://site.example/page", summary="", snippet="", authority=-1):
    return {"title": title, "url": url, "summary": summary, "snippet": snippet, "authority": authority}


def _row(rid, answer, results, *, question="q", error=None, latency=0.3, expected=()):
    return {
        "id": rid,
        "question": question,
        "answer": answer,
        "expected_urls": list(expected),
        "error": error,
        "latency": latency,
        "results": results,
    }


def _run(rows, **meta):
    return {"meta": {"set": "dev", "label": "t", "count": 50, "created_utc": "2026-10-21", **meta}, "rows": rows}


class TestMatching:
    def test_norm(self):
        assert score.norm("3,000 Café — Ünïcode!") == "3000 cafe unicode"

    @pytest.mark.parametrize(
        ("answer", "text", "expected"),
        [
            ("2007", "He was sentenced in 2007.", True),
            ("3,000", "about 3000 people", True),
            ("William Meny and Paul Simms", "written by Paul Simms and William Meny", True),
            ("William Meny and Paul Simms", "written by Paul Simms", False),
            ("the", "the the", True),  # exact phrase still counts
            ("b c d e f g h", "h g f e d c b", False),  # more than 6 significant words: exact phrase only
            ("a b c d e f g", "g f e d c b", True),  # "a" is a stop word, so only 6 words count
            ("", "anything", False),
            ("Odessa, Florida", "born in Odessa (Florida)", True),
        ],
    )
    def test_matches(self, answer, text, expected):
        assert score.matches(answer, text) is expected


class TestRanking:
    @pytest.mark.parametrize(
        ("url", "excluded"),
        [
            ("https://huggingface.co/datasets/basicv8vc/SimpleQA", True),
            ("https://huggingface.co/api/resolve-cache/datasets/x", True),
            ("https://www.kaggle.com/datasets/simpleqa", True),
            ("https://github.com/openai/simple-evals/blob/main/x.csv", True),
            ("https://github.com/other/repo", False),
            ("https://en.wikipedia.org/wiki/Simple", False),
            ("huggingface.co/datasets/x", True),
        ],
    )
    def test_is_excluded(self, url, excluded):
        assert score.is_excluded(url) is excluded

    def test_exclusion_shifts_ranks(self):
        results = [
            _r(url="https://huggingface.co/datasets/x", summary="answer 1869"),
            _r(summary="nothing"),
            _r(summary="founded 1869"),
        ]
        assert score.first_hit("1869", results) == 2
        assert score.first_hit("1869", results, exclude=False) == 1

    def test_url_and_title_count(self):
        assert score.first_hit("Moog", [_r(url="https://x.example/robert_moog")]) == 1
        assert score.first_hit("Felix Chen", [_r(title="Felix Chen obituary")]) == 1
        assert score.first_hit("Felix Chen", []) is None

    @pytest.mark.parametrize(
        ("got", "rank"),
        [
            ("https://www.sozo.id/lp/", 1),
            ("http://sozo.id/lp/radiofrequency?x=1", 1),
            ("https://sozo.id/lp2", None),
            ("https://sozo.sg/lp", None),
        ],
    )
    def test_url_hit(self, got, rank):
        assert score.url_hit(["sozo.id/lp"], [_r(url=got)]) == rank

    def test_rank_of_prefers_expected_urls(self):
        row = _row("1", "irrelevant", [], expected=("https://a.example/x",))
        assert score.rank_of(row, [_r(url="https://b.example"), _r(url="https://a.example/x/y")]) == 2
        assert score.rank_of(_row("2", "irrelevant", []), [_r(summary="irrelevant")]) == 1

    def test_score_run_top(self):
        results = [_r(summary="no")] * 11 + [_r(summary="answer here")]
        run = _run([_row("1", "answer here", results)])
        assert score.score_run(run) == {"1": 12}
        assert score.score_run(run, top=10) == {"1": None}


class TestStatistics:
    def test_bootstrap_deterministic(self):
        flags = [True] * 62 + [False] * 38
        first = score.bootstrap(flags)
        assert first == score.bootstrap(flags)
        assert first[0] == 0.62
        assert first[1] < 0.62 < first[2]
        assert score.bootstrap([]) == (0.0, 0.0, 0.0)

    def test_mcnemar_matches_original_analysis(self):
        a = [True] * 53 + [False] * 10 + [True] * 100
        b = [False] * 53 + [True] * 10 + [True] * 100
        only_a, only_b, p = score.mcnemar(a, b)
        assert (only_a, only_b) == (53, 10)
        assert p == pytest.approx(3.382168502403561e-08)

    def test_mcnemar_no_discordant_pairs(self):
        assert score.mcnemar([True, False], [True, False]) == (0, 0, 1.0)

    def test_metrics(self):
        long_q = "x" * 129
        run = _run(
            [
                _row("1", "a1", [_r(summary="a1")], latency=0.2),
                _row("2", "a2", [_r(summary="no"), _r(summary="a2")], latency=0.4, question=long_q),
                _row("3", "a3", [], latency=0.6),
                _row("4", "a4", [], error="HTTP 401", latency=0.1),
            ]
        )
        m = score.metrics(run, score.score_run(run))
        assert m.n == 4
        assert m.hits == {1: 1, 3: 2, 5: 2, 10: 2}
        assert m.mrr == pytest.approx((1 + 0.5) / 4)
        assert (m.errors, m.empty, m.billable) == (1, 1, 2)
        assert m.cost == pytest.approx(2 * score.PRICE_PER_REQUEST)
        assert (m.p50, m.p90) == (0.4, 0.6)
        assert (m.long_n, m.long_hits10) == (1, 1)

    def test_metrics_empty_run(self):
        m = score.metrics(_run([]), {})
        assert (m.n, m.mrr, m.p50) == (0, 0.0, None)


class TestReports:
    def test_summary(self):
        run = _run([_row("1", "a", [_r(summary="a")]), _row("2", "b", [_r(summary="x")], question="y" * 200)])
        text = score.summary(run, top=10)
        assert "set dev · label t · 2 questions · count 50" in text
        assert "scored on top 10" in text
        assert "@1    50.0%" in text
        assert "long (>128 chars) 1 questions, @10 0.0%" in text

    def test_summary_without_latency_or_long_rows(self):
        text = score.summary(_run([_row("1", "a", [], error="boom", latency=None)]))
        assert "latency n/a" in text
        assert "@10 n/a" in text

    def test_compare_pairs_by_id(self):
        a = _run([_row("1", "a", [_r(summary="x")]), _row("2", "b", [_r(summary="b")]), _row("3", "c", [])])
        b = _run([_row("2", "b", [_r(summary="b")]), _row("1", "a", [_r(summary="a")])], label="cand")
        text = score.compare(a, b)
        assert "paired questions 2, skipped 1" in text
        assert "only A 0, only B 1" in text

    def test_compare_without_shared_ids(self):
        text = score.compare(_run([_row("1", "a", [])]), _run([_row("2", "a", [])]))
        assert "paired questions 0, skipped 2" in text


class TestCli:
    def test_score_and_compare(self, tmp_path, capsys):
        path = tmp_path / "a.json"
        path.write_text(json.dumps(_run([_row("1", "a", [_r(summary="a")])])), encoding="utf-8")
        assert score.main([str(path)]) == 0
        assert "@1   100.0%" in capsys.readouterr().out
        assert score.main([str(path), "--compare", str(path)]) == 0
        assert "McNemar p=1" in capsys.readouterr().out

    def test_missing_file(self, tmp_path, capsys):
        assert score.main([str(tmp_path / "nope.json")]) == 2
        assert "Cannot read run" in capsys.readouterr().err
