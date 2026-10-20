"""Score benchmark runs: rank of the first correct result, @k, MRR, bootstrap CIs and paired McNemar tests.

A SimpleQA row is correct at rank N when the normalised answer text appears in the title, URL or summary/snippet of
result N (the owner's ``matches`` rule, kept verbatim). Pages of the SimpleQA dataset itself are removed first so an
engine cannot "find" the answer key. Rows with ``expected_urls`` (local sets) are scored by URL instead.

Run: ``uv run python -m bench.score <run.json> [--compare <other run.json>] [--top 10]``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PRICE_PER_REQUEST = 0.00075  # USD per successful, non-empty Search request
KS = (1, 3, 5, 10)  # Hermes web_search shows 5 results by default
LONG_CHARS = 128  # DeepSeek's query limit
EXCLUDED_HOSTS = ("huggingface.co", "kaggle.com")  # dataset mirrors that contain the answer key
EXCLUDED_PREFIXES = ("github.com/openai/simple-evals",)

# --- The owner's matching rule, copied verbatim from the original run_benchmark.py ---------------------------------
STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "de", "la", "le", "von", "van"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"(?<=\d)[,.](?=\d{3}\b)", "", s)  # 3,000 -> 3000
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def matches(answer: str, text: str) -> bool:
    a, t = norm(answer), " " + norm(text) + " "
    if not a:
        return False
    if " " + a + " " in t:
        return True
    toks = [w for w in a.split() if w not in STOP]
    return 1 < len(toks) <= 6 and all(" " + w + " " in t for w in toks)  # every word in the same result


# --- Ranking ------------------------------------------------------------------------------------------------------


def is_excluded(url: str) -> bool:
    parts = urlsplit(url if "://" in url else "//" + url)
    host = (parts.hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in EXCLUDED_HOSTS):
        return True
    target = host.removeprefix("www.") + parts.path.lower()
    return any(target.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def result_text(result: Mapping[str, Any]) -> str:
    url = str(result.get("url") or "").replace("_", " ").replace("-", " ")
    parts = (result.get("title"), url, result.get("summary"), result.get("snippet"))
    return " ".join(str(p) for p in parts if p)


def first_hit(answer: str, results: Sequence[Mapping[str, Any]], *, exclude: bool = True) -> int | None:
    kept = [r for r in results if not (exclude and is_excluded(str(r.get("url") or "")))]
    for rank, result in enumerate(kept, start=1):
        if matches(answer, result_text(result)):
            return rank
    return None


def _norm_url(url: str) -> str:
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", url.strip().lower())
    return u.removeprefix("www.").rstrip("/")


def url_hit(expected_urls: Sequence[str], results: Sequence[Mapping[str, Any]]) -> int | None:
    """Rank of the first result whose URL is one of ``expected_urls`` or a page below it."""
    targets = [_norm_url(u) for u in expected_urls if u.strip()]
    for rank, result in enumerate(results, start=1):
        got = _norm_url(str(result.get("url") or ""))
        if any(got == t or got.startswith((t + "/", t + "?", t + "#")) for t in targets):
            return rank
    return None


def rank_of(row: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> int | None:
    expected = row.get("expected_urls") or ()
    if expected:
        return url_hit(expected, results)
    return first_hit(str(row.get("answer") or ""), results)


def score_run(run: Mapping[str, Any], *, top: int | None = None) -> dict[str, int | None]:
    """``{row id: rank}``; ``top`` keeps only the first N results (what a ``count=N`` caller would have seen)."""
    ranks: dict[str, int | None] = {}
    for row in run["rows"]:
        results = row.get("results") or []
        ranks[row["id"]] = rank_of(row, results[:top] if top else results)
    return ranks


# --- Statistics -----------------------------------------------------------------------------------------------------


def hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def bootstrap(flags: Sequence[bool], *, samples: int = 5000, seed: int = 7) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval of a hit rate."""
    n = len(flags)
    if not n:
        return 0.0, 0.0, 0.0
    rnd = random.Random(seed)
    means = sorted(sum(flags[rnd.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return sum(flags) / n, means[int(0.025 * samples)], means[int(0.975 * samples)]


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired outcomes → ``(only a, only b, p)``."""
    b10 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b01 = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = b10 + b01
    if not n:
        return b10, b01, 1.0
    p = 2 * sum(math.comb(n, j) for j in range(min(b10, b01) + 1)) / 2**n
    return b10, b01, min(1.0, p)


@dataclass(frozen=True, slots=True)
class Metrics:
    n: int
    hits: dict[int, int]
    mrr: float
    errors: int
    empty: int
    billable: int
    cost: float
    p50: float | None
    p90: float | None
    long_n: int
    long_hits10: int


def metrics(run: Mapping[str, Any], ranks: Mapping[str, int | None]) -> Metrics:
    rows = run["rows"]
    n = len(rows)
    ok = [r for r in rows if not r.get("error")]
    latencies = sorted(float(r["latency"]) for r in ok if r.get("latency") is not None)
    billable = sum(1 for r in ok if r.get("results"))
    long_rows = [r for r in rows if len(str(r.get("question") or "")) > LONG_CHARS]
    return Metrics(
        n=n,
        hits={k: sum(1 for r in rows if hit(ranks.get(r["id"]), k)) for k in KS},
        mrr=sum(1 / ranks[r["id"]] for r in rows if ranks.get(r["id"])) / n if n else 0.0,
        errors=n - len(ok),
        empty=sum(1 for r in ok if not r.get("results")),
        billable=billable,
        cost=billable * PRICE_PER_REQUEST,
        p50=latencies[len(latencies) // 2] if latencies else None,
        p90=latencies[int(len(latencies) * 0.9)] if latencies else None,
        long_n=len(long_rows),
        long_hits10=sum(1 for r in long_rows if hit(ranks.get(r["id"]), 10)),
    )


# --- Reports --------------------------------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _describe(run: Mapping[str, Any]) -> str:
    meta = run.get("meta") or {}
    return (
        f"set {meta.get('set', '?')} · label {meta.get('label', '?')} · {len(run['rows'])} questions · "
        f"count {meta.get('count', '?')} · {meta.get('created_utc', '?')}"
    )


def summary(run: Mapping[str, Any], *, top: int | None = None) -> str:
    ranks = score_run(run, top=top)
    m = metrics(run, ranks)
    ids = [r["id"] for r in run["rows"]]
    lines = [_describe(run) + (f" · scored on top {top}" if top else "")]
    for k in KS:
        mean, lo, hi = bootstrap([hit(ranks[i], k) for i in ids])
        lines.append(f"  @{k:<3} {_pct(mean)}  [{_pct(lo).strip()} – {_pct(hi).strip()}]  ({m.hits[k]}/{m.n})")
    long_rate = f"{_pct(m.long_hits10 / m.long_n).strip()}" if m.long_n else "n/a"
    latency = f"p50 {m.p50:.2f}s p90 {m.p90:.2f}s" if m.p50 is not None else "n/a"
    lines.append(f"  MRR {m.mrr:.3f} · long (>{LONG_CHARS} chars) {m.long_n} questions, @10 {long_rate}")
    lines.append(f"  errors {m.errors} · empty {m.empty} · latency {latency} · cost ${m.cost:.4f}")
    return "\n".join(lines)


def compare(run_a: Mapping[str, Any], run_b: Mapping[str, Any], *, top: int | None = None) -> str:
    """Paired comparison on the questions both runs answered (by id); B is the candidate."""
    ranks_a, ranks_b = score_run(run_a, top=top), score_run(run_b, top=top)
    ids = [i for i in ranks_a if i in ranks_b]
    skipped = len(set(ranks_a) ^ set(ranks_b))
    lines = [f"A: {_describe(run_a)}", f"B: {_describe(run_b)}", f"  paired questions {len(ids)}, skipped {skipped}"]
    for k in KS:
        a = [hit(ranks_a[i], k) for i in ids]
        b = [hit(ranks_b[i], k) for i in ids]
        only_a, only_b, p = mcnemar(a, b)
        delta = (sum(b) - sum(a)) / len(ids) if ids else 0.0
        lines.append(
            f"  @{k:<3} A {_pct(sum(a) / len(ids) if ids else 0)}  B {_pct(sum(b) / len(ids) if ids else 0)}  "
            f"Δ {100 * delta:+.1f} pts  only A {only_a}, only B {only_b}, McNemar p={p:.3g}"
        )
    return "\n".join(lines)


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utf8_stdout() -> None:
    """Report symbols (·, Δ, –) must not crash a cp1252 Windows console."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m bench.score", description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="run JSON written by bench.run")
    parser.add_argument("--compare", type=Path, help="candidate run JSON to compare against RUN (paired by id)")
    parser.add_argument("--top", type=int, help="score only the first N results of each query")
    args = parser.parse_args(argv)
    try:
        run = load_run(args.run)
        other = load_run(args.compare) if args.compare else None
    except (OSError, ValueError) as exc:
        print(f"Cannot read run: {exc}", file=sys.stderr)
        return 2
    print(summary(run, top=args.top))
    if other is not None:
        print()
        print(summary(other, top=args.top))
        print()
        print(compare(run, other, top=args.top))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
