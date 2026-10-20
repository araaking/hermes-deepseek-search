"""Tests for bench.variants.

Covers:
- query_terms (question words dropped)
- rerank (overlap wins, zero weights keep order, top_k cut, authority, order prior)
- evaluate (reference = DeepSeek order over all results) and the grid
- report layout and the CLI
"""

from __future__ import annotations

import json

from bench import variants
from bench.variants import Variant


def _r(title, summary="", authority=-1):
    slug = title.lower().replace(" ", "-")
    return {"title": title, "url": f"https://{slug}.example", "summary": summary, "snippet": "", "authority": authority}


def _run(rows):
    return {"meta": {"set": "dev"}, "rows": rows}


QUESTION = "In which city did Ciara perform on her Jackie Tour?"
RESULTS = [_r("Unrelated page"), _r("Other"), _r("Ciara Jackie Tour", summary="Ciara performed in Riverside")]


def test_query_terms_drop_question_words():
    assert variants.query_terms(QUESTION) == frozenset({"city", "ciara", "perform", "jackie", "tour"})


def test_rerank_moves_overlap_to_top():
    assert variants.rerank(QUESTION, RESULTS, Variant(10))[0]["title"] == "Ciara Jackie Tour"


def test_zero_weights_keep_deepseek_order():
    order = variants.rerank(QUESTION, RESULTS, Variant(10, w_overlap=0.0))
    assert [r["title"] for r in order] == [r["title"] for r in RESULTS]


def test_top_k_cuts_the_pool():
    assert len(variants.rerank(QUESTION, RESULTS, Variant(2))) == 2


def test_authority_and_order_prior():
    tied = [_r("A", authority=0), _r("B", authority=4), _r("C", authority="bad")]
    assert variants.rerank("zzz", tied, Variant(10, w_overlap=0.0, w_authority=1.0))[0]["title"] == "B"
    assert variants.rerank("zzz", tied, Variant(10, w_overlap=0.0, w_order=1.0))[0]["title"] == "A"


def test_grid():
    grid = variants.grid()
    assert len(grid) == len(variants.TOP_KS) * len(variants.W_AUTHORITY) * len(variants.W_ORDER)
    assert grid[0].label() == "top10 overlap=1 authority=0 order=0"


def test_evaluate_reference_uses_all_results():
    results = [_r(f"Page {i}") for i in range(12)] + [_r("Hit", summary="Riverside")]
    run = _run([{"id": "1", "question": QUESTION, "answer": "Riverside", "results": results}])
    ranks = variants.evaluate(run, [Variant(10)])
    assert ranks[None] == {"1": 13}
    assert ranks[Variant(10)] == {"1": None}  # outside the reranked top 10


def test_report_layout():
    run = _run([{"id": "1", "question": QUESTION, "answer": "Riverside", "results": RESULTS}])
    lines = variants.report(run, best=2, variants=[Variant(10), Variant(10, w_overlap=0.0)]).splitlines()
    assert lines[0].startswith("1 questions · best 2 of 2 variants")
    assert "uncorrected" in lines[1]
    assert "confirm a pick on the test set" in lines[1]
    assert lines[2].endswith("DeepSeek order (reference)")
    assert "@1 100.0%" in lines[3]
    assert "(@1 vs DeepSeek: +1/-0" in lines[3]


def test_cli(tmp_path, capsys):
    path = tmp_path / "run.json"
    run = _run([{"id": "1", "question": QUESTION, "answer": "x", "results": RESULTS}])
    path.write_text(json.dumps(run), encoding="utf-8")
    assert variants.main([str(path), "--best", "1"]) == 0
    assert "of 64 variants" in capsys.readouterr().out
    assert variants.main([str(tmp_path / "missing.json")]) == 2
    assert "Cannot read run" in capsys.readouterr().err
