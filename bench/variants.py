"""Offline rerank / "fetch N" experiments on one benchmark run — no API calls, no cost.

DeepSeek already orders results by its own ``rerank_score``, so a rerank only helps with signals of our own: how
many query terms a result contains (title, summary, snippet), its ``authority`` (0–4), and DeepSeek's position as a
prior. Each variant reranks the first ``top_k`` results (10/20/30/50: "fetch N") and is scored with ``bench.score``.

Tune on the dev set only; use the test and long sets once, to confirm the chosen variant. This module is the
prototype for accuracy Phase 3, which moves the winner into ``core``.

Run: ``uv run python -m bench.variants <run.json> [--best 10]``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench import score

QSTOP = score.STOP | set(
    "what which who whom whose when where why how is was were are did does do year day month name according please "
    "give by with from as that this its it his her their be been".split()
)
TOP_KS = (10, 20, 30, 50)
W_AUTHORITY = (0.0, 0.1, 0.25, 0.5)
W_ORDER = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class Variant:
    top_k: int
    w_overlap: float = 1.0
    w_authority: float = 0.0
    w_order: float = 0.0

    def label(self) -> str:
        return f"top{self.top_k} overlap={self.w_overlap:g} authority={self.w_authority:g} order={self.w_order:g}"


def grid() -> list[Variant]:
    return [Variant(k, 1.0, a, o) for k, a, o in itertools.product(TOP_KS, W_AUTHORITY, W_ORDER)]


def query_terms(question: str) -> frozenset[str]:
    return frozenset(w for w in score.norm(question).split() if w not in QSTOP and len(w) > 1)


def result_tokens(result: Mapping[str, Any]) -> frozenset[str]:
    text = " ".join(str(result.get(f) or "") for f in ("title", "summary", "snippet"))
    return frozenset(score.norm(text).split())


def _authority(result: Mapping[str, Any]) -> float:
    try:
        return max(int(result.get("authority", -1)), 0) / 4
    except (TypeError, ValueError):
        return 0.0


def _rerank(
    terms: frozenset[str],
    tokens: Sequence[frozenset[str]],
    results: Sequence[Mapping[str, Any]],
    variant: Variant,
) -> list[Mapping[str, Any]]:
    def key(index: int) -> float:
        overlap = len(terms & tokens[index]) / max(1, len(terms))
        return (
            variant.w_overlap * overlap
            + variant.w_authority * _authority(results[index])
            + variant.w_order / (index + 1)
        )

    order = sorted(range(min(variant.top_k, len(results))), key=key, reverse=True)  # stable for ties
    return [results[i] for i in order]


def rerank(question: str, results: Sequence[Mapping[str, Any]], variant: Variant) -> list[Mapping[str, Any]]:
    """The first ``variant.top_k`` results, reordered by the variant's weighted score."""
    return _rerank(query_terms(question), [result_tokens(r) for r in results], results, variant)


def evaluate(run: Mapping[str, Any], variants: Sequence[Variant]) -> dict[Variant | None, dict[str, int | None]]:
    """``{variant: {row id: rank}}``; the ``None`` key is DeepSeek's own order over all stored results."""
    prepared = [
        (row, query_terms(str(row.get("question") or "")), [result_tokens(r) for r in row.get("results") or []])
        for row in run["rows"]
    ]
    out: dict[Variant | None, dict[str, int | None]] = {
        None: {row["id"]: score.rank_of(row, row.get("results") or []) for row, _, _ in prepared}
    }
    for variant in variants:
        out[variant] = {
            row["id"]: score.rank_of(row, _rerank(terms, tokens, row.get("results") or [], variant))
            for row, terms, tokens in prepared
        }
    return out


def report(run: Mapping[str, Any], *, best: int = 10, variants: Sequence[Variant] | None = None) -> str:
    ranks = evaluate(run, grid() if variants is None else variants)
    ids = [row["id"] for row in run["rows"]]
    n = len(ids) or 1
    reference = [score.hit(ranks[None][i], 1) for i in ids]

    def line(name: str, by_id: Mapping[str, int | None], vs_reference: bool) -> str:
        cells = "  ".join(f"@{k} {100 * sum(score.hit(by_id[i], k) for i in ids) / n:5.1f}%" for k in score.KS)
        if not vs_reference:
            return f"  {cells}  {name}"
        only_ref, only_variant, p = score.mcnemar(reference, [score.hit(by_id[i], 1) for i in ids])
        return f"  {cells}  {name}  (@1 vs DeepSeek: +{only_variant}/-{only_ref}, p={p:.3g})"

    def sort_key(variant: Variant) -> tuple[int, ...]:
        return tuple(sum(score.hit(ranks[variant][i], k) for i in ids) for k in (1, 5, 10))

    ranked = sorted((v for v in ranks if v is not None), key=sort_key, reverse=True)
    lines = [
        f"{len(ids)} questions · best {min(best, len(ranked))} of {len(ranked)} variants by @1 (then @5, @10)",
        line("DeepSeek order (reference)", ranks[None], vs_reference=False),
    ]
    lines += [line(v.label(), ranks[v], vs_reference=True) for v in ranked[:best]]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    score.utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m bench.variants", description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="run JSON written by bench.run (count=50 recommended)")
    parser.add_argument("--best", type=int, default=10, help="how many variants to show")
    args = parser.parse_args(argv)
    try:
        run = score.load_run(args.run)
    except (OSError, ValueError) as exc:
        print(f"Cannot read run: {exc}", file=sys.stderr)
        return 2
    print(report(run, best=args.best))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
