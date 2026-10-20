"""End-to-end smoke test against a real Hermes Agent install.

Run with Hermes' own interpreter so its modules are importable::

    ~/.hermes/hermes-agent/venv/bin/python -m hermes_deepseek_search.smoke            # isolated (default)
    ~/.hermes/hermes-agent/venv/bin/python -m hermes_deepseek_search.smoke --current-home

Isolated mode points ``HERMES_HOME`` at a throwaway directory whose config selects ``deepseek``, enables the
``web-deepseek`` plugin and turns Hermes' keyless rescue OFF — otherwise a failing provider is silently answered
by another vendor. Live checks cost about 3 successful requests (~$0.00225); failed requests are not billed.
Exit codes: 0 all passed, 1 a check failed, 2 Hermes is not importable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from hermes_deepseek_search import __version__
from hermes_deepseek_search.core import API_KEY_ENV

PLUGIN_ID = "web-deepseek"
PROVIDER_NAME = "deepseek"
_INVALID_KEY = "sk-invalid-hermes-deepseek-smoke"
_LONG_QUERY = " ".join(["deepseek"] * 20)  # 179 chars → forces the 128-char truncation notice
_METADATA_MARK = "relevance "  # every DeepSeek row ends with "[… relevance x.y/3 …]"

Status = Literal["PASS", "FAIL", "SKIP"]


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str = ""

    def line(self) -> str:
        return f"[{self.status}] {self.name}" + (f" — {self.detail}" if self.detail else "")


def isolated_config_text() -> str:
    """config.yaml for the throwaway home: select deepseek for both tools, enable the plugin, no rescue."""
    return (
        "web:\n"
        f"  backend: {PROVIDER_NAME}\n"
        "  keyless_fallback: false\n"
        "  keyless_rescue: false\n"
        "plugins:\n"
        "  enabled:\n"
        f"    - {PLUGIN_ID}\n"
    )


def _rescued(payload: dict[str, Any]) -> str:
    """Non-empty description when Hermes answered with another backend instead of deepseek."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    served, origin = data.get("served_by"), data.get("rescued_from")
    if not (served or origin):
        return ""
    return f"served by {served!r} (rescued from {origin!r}): {data.get('backend_error', '')}".strip()


def evaluate_search(payload: dict[str, Any], name: str = "live web_search") -> Check:
    if payload.get("success") is not True:
        return Check(name, "FAIL", f"error: {payload.get('error')}")
    rescued = _rescued(payload)
    if rescued:
        return Check(name, "FAIL", rescued)
    web = (payload.get("data") or {}).get("web")
    if not isinstance(web, list) or not web:
        return Check(name, "FAIL", "no results returned")
    required = ("title", "url", "description", "position")
    if any(not isinstance(r, dict) or any(k not in r for k in required) for r in web):
        return Check(name, "FAIL", f"a result row is missing one of {required}")
    if not any(_METADATA_MARK in str(r["description"]) for r in web):
        return Check(name, "FAIL", "descriptions lack the DeepSeek metadata line — another backend served this")
    return Check(name, "PASS", f"{len(web)} results; top: {str(web[0]['title'])[:70]!r}")


def evaluate_extract(
    payload: dict[str, Any], name: str = "live web_extract", *, rescue_possible: bool = False
) -> Check:
    if payload.get("success") is False:
        return Check(name, "FAIL", f"error: {payload.get('error')}")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return Check(name, "FAIL", "no results returned")
    first = results[0] if isinstance(results[0], dict) else {}
    if first.get("error"):
        return Check(name, "FAIL", f"error: {first['error']}")
    content = str(first.get("content") or "")
    if not content:
        return Check(name, "FAIL", "empty content")
    note = " (source unverified: keyless rescue may be on)" if rescue_possible else ""
    return Check(name, "PASS", f"{len(content):,} chars{note}")


def evaluate_search_error(payload: dict[str, Any], needle: str, name: str) -> Check:
    rescued = _rescued(payload)
    if rescued:
        return Check(name, "FAIL", f"expected a deepseek error but {rescued}")
    error = str(payload.get("error", ""))
    if payload.get("success") is False and needle in error:
        return Check(name, "PASS", error[:120])
    return Check(name, "FAIL", f"expected an error containing {needle!r}, got {json.dumps(payload)[:200]}")


def evaluate_extract_error(payload: dict[str, Any], needle: str, name: str) -> Check:
    results = payload.get("results")
    first = results[0] if isinstance(results, list) and results and isinstance(results[0], dict) else {}
    error = str(first.get("error") or payload.get("error") or "")
    if needle in error:
        return Check(name, "PASS", error[:120])
    return Check(name, "FAIL", f"expected an error containing {needle!r}, got {json.dumps(payload)[:200]}")


def evaluate_notice(payload: dict[str, Any], name: str = "long-query notice") -> Check:
    if payload.get("success") is True and not _rescued(payload) and "notice" in payload:
        return Check(name, "PASS", str(payload["notice"])[:120])
    return Check(name, "FAIL", f"no 'notice' in response: {json.dumps(payload)[:200]}")


def summarize(checks: Sequence[Check]) -> tuple[str, int]:
    """``("N passed, M failed, K skipped", exit_code)``; exit 1 when anything failed."""
    counts = {s: sum(c.status == s for c in checks) for s in ("PASS", "FAIL", "SKIP")}
    text = f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped"
    return text, (1 if counts["FAIL"] else 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m hermes_deepseek_search.smoke", description=__doc__.splitlines()[0])
    p.add_argument(
        "--current-home", action="store_true", help="test the real HERMES_HOME/config instead of a throwaway home"
    )
    p.add_argument(
        "--register", action="store_true", help="register the provider directly instead of via entry-point discovery"
    )
    p.add_argument("--skip-live", action="store_true", help="skip billable live checks (search/extract/notice)")
    p.add_argument("--quick", action="store_true", help="skip the long-query notice check (one request fewer)")
    p.add_argument("--query", default="DeepSeek API release notes", help="live search query")
    p.add_argument("--url", default="https://example.com", help="live extract URL")
    p.add_argument("--keep-home", action="store_true", help="do not delete the isolated home afterwards")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        import tools.web_tools  # noqa: F401
        from agent import web_search_registry  # noqa: F401
    except ImportError as exc:
        print(
            f"Hermes Agent is not importable ({exc}). Run this with Hermes' own interpreter, e.g. "
            "~/.hermes/hermes-agent/venv/bin/python -m hermes_deepseek_search.smoke"
        )
        return 2
    return _report(args)


# --- Everything below drives a real Hermes install; exercised by the pre-flight, not unit tests. -----------------


def _report(args: argparse.Namespace) -> int:  # pragma: no cover
    mode = "current home" if args.current_home else "isolated home (keyless rescue off)"
    print(f"Hermes Agent {_hermes_version()} · hermes-deepseek-search {__version__} · {mode}")
    checks = run(args)
    for check in checks:
        print(check.line())
    text, code = summarize(checks)
    print(text)
    return code


def _hermes_version() -> str:  # pragma: no cover
    try:
        import hermes_cli

        return str(getattr(hermes_cli, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _resolve_key_from_real_home() -> str:  # pragma: no cover
    """DEEPSEEK_API_KEY as Hermes sees it in the REAL home (env, then <home>/.env), read in a child process so
    no config cache leaks into the isolated run. Never printed."""
    code = f"from agent.web_search_provider import get_provider_env as g; print(g({API_KEY_ENV!r}))"
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = out.stdout.strip().splitlines()
    return lines[-1].strip() if lines else ""


def run(args: argparse.Namespace) -> list[Check]:  # pragma: no cover
    isolated = not args.current_home
    if isolated and not os.environ.get(API_KEY_ENV, "").strip():
        key = _resolve_key_from_real_home()
        if key:
            os.environ[API_KEY_ENV] = key
    home = None
    if isolated:
        home = tempfile.mkdtemp(prefix="deepseek-smoke-")
        with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
            fh.write(isolated_config_text())
        os.environ["HERMES_HOME"] = home
    try:
        return _run_checks(args, isolated=isolated)
    finally:
        if home and not args.keep_home:
            shutil.rmtree(home, ignore_errors=True)


def _run_checks(args: argparse.Namespace, *, isolated: bool) -> list[Check]:  # pragma: no cover
    from agent import web_search_registry as registry
    from tools.web_tools import web_extract_tool, web_search_tool

    from hermes_deepseek_search.hermes._sync import run_sync

    if args.register:
        from hermes_deepseek_search.hermes.provider import DeepSeekWebSearchProvider

        registry.register_provider(DeepSeekWebSearchProvider())
    checks: list[Check] = []
    key = os.environ.get(API_KEY_ENV, "").strip()

    def search(query: str) -> dict[str, Any]:
        return json.loads(web_search_tool(query, limit=5))

    def extract(url: str) -> dict[str, Any]:
        return json.loads(run_sync(lambda: web_extract_tool([url])))

    # The first tool call triggers Hermes plugin discovery; error-path calls are free (nothing billable succeeds).
    missing_search: dict[str, Any] = {}
    missing_extract: dict[str, Any] = {}
    if isolated:
        os.environ.pop(API_KEY_ENV, None)
        missing_search = search("deepseek smoke missing key")
        missing_extract = extract(args.url)
    provider = registry.get_provider(PROVIDER_NAME)
    if provider is None:
        checks.append(
            Check(
                "plugin registered",
                "FAIL",
                f"no '{PROVIDER_NAME}' provider — install the package into Hermes' venv and add "
                f"'{PLUGIN_ID}' to plugins.enabled (or pass --register)",
            )
        )
        return checks
    how = "registered directly (--register)" if args.register else "via entry point"
    checks.append(Check("plugin registered", "PASS", f"{PROVIDER_NAME} {how}"))

    if isolated:
        needle = f"{API_KEY_ENV} is not set"
        checks.append(evaluate_search_error(missing_search, needle, "missing-key error path (search)"))
        checks.append(evaluate_extract_error(missing_extract, needle, "missing-key error path (extract)"))
        os.environ[API_KEY_ENV] = _INVALID_KEY
        checks.append(evaluate_search_error(search("deepseek smoke invalid key"), "HTTP 401", "invalid-key error path"))
        if key:
            os.environ[API_KEY_ENV] = key
        else:
            os.environ.pop(API_KEY_ENV, None)

    if not key:
        checks.append(Check("API key available", "SKIP", f"{API_KEY_ENV} not set — live checks skipped"))
        return checks
    checks.append(Check("API key available", "PASS", "present (value not shown)"))
    if args.skip_live:
        checks.append(Check("live checks", "SKIP", "--skip-live"))
        return checks
    checks.append(evaluate_search(search(args.query)))
    checks.append(evaluate_extract(extract(args.url), rescue_possible=not isolated))
    if not args.quick:
        checks.append(evaluate_notice(search(_LONG_QUERY)))
    return checks


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
