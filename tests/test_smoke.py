"""Tests for the Hermes smoke-test evaluators and CLI (the Hermes-driving runner is covered by the pre-flight).

Covers:
- evaluate_search — pass, keyless-rescue detection, failure, empty, incomplete rows, missing DeepSeek metadata
- evaluate_extract — pass (with/without rescue caveat), per-URL error, empty, envelope failure
- evaluate_search_error / evaluate_extract_error — expected message, rescued response, wrong message
- evaluate_notice — present, absent, rescued
- summarize / Check.line — counts, exit codes, formatting
- isolated_config_text — selects deepseek, enables the plugin, disables rescue
- parse_args — defaults and every flag
- main — exits 2 with guidance when Hermes is not importable
"""

from __future__ import annotations

import sys
import types

import pytest

from hermes_deepseek_search import smoke
from hermes_deepseek_search.smoke import (
    Check,
    evaluate_extract,
    evaluate_extract_error,
    evaluate_notice,
    evaluate_search,
    evaluate_search_error,
    isolated_config_text,
    parse_args,
    summarize,
)

_ROW = {
    "title": "DeepSeek V4 model released",
    "url": "https://example.com/deepseek-v4",
    "description": "Summary.\n[Example News · 2026-06-01 · relevance 2.4/3 · authority 3/4]",
    "position": 1,
}
_RESCUED = {
    "success": True,
    "data": {
        "web": [{"title": "t", "url": "u", "description": "exa text", "position": 1}],
        "served_by": "exa",
        "rescued_from": "deepseek",
        "backend_error": "Configured backend 'deepseek' failed this call (DEEPSEEK_API_KEY is not set)",
    },
}

# ---------------------------------------------------------------------------
# evaluate_search
# ---------------------------------------------------------------------------


class TestEvaluateSearch:
    def test_pass(self):
        check = evaluate_search({"success": True, "data": {"web": [_ROW, {**_ROW, "position": 2}]}})
        assert check.status == "PASS"
        assert check.detail.startswith("2 results; top: 'DeepSeek V4 model released'")

    def test_rescued_response_fails(self):
        check = evaluate_search(_RESCUED)
        assert check.status == "FAIL"
        assert "'exa'" in check.detail
        assert "DEEPSEEK_API_KEY is not set" in check.detail

    def test_failure_envelope(self):
        check = evaluate_search({"success": False, "error": "boom"})
        assert (check.status, check.detail) == ("FAIL", "error: boom")

    @pytest.mark.parametrize("payload", [{"success": True, "data": {"web": []}}, {"success": True}])
    def test_no_results(self, payload):
        check = evaluate_search(payload)
        assert check.status == "FAIL"
        assert "no results" in check.detail

    def test_incomplete_row(self):
        row = {k: v for k, v in _ROW.items() if k != "position"}
        check = evaluate_search({"success": True, "data": {"web": [row]}})
        assert check.status == "FAIL"
        assert "missing one of" in check.detail

    def test_missing_deepseek_metadata(self):
        row = {**_ROW, "description": "plain text from another vendor"}
        check = evaluate_search({"success": True, "data": {"web": [row]}})
        assert check.status == "FAIL"
        assert "another backend" in check.detail


# ---------------------------------------------------------------------------
# evaluate_extract
# ---------------------------------------------------------------------------


class TestEvaluateExtract:
    def test_pass(self):
        check = evaluate_extract({"results": [{"url": "u", "content": "abc", "error": None}]})
        assert (check.status, check.detail) == ("PASS", "3 chars")

    def test_pass_with_rescue_caveat(self):
        check = evaluate_extract({"results": [{"content": "abc"}]}, rescue_possible=True)
        assert check.status == "PASS"
        assert "unverified" in check.detail

    def test_per_url_error(self):
        check = evaluate_extract({"results": [{"content": "", "error": "DeepSeek Fetch returned HTTP 404"}]})
        assert check.status == "FAIL"
        assert "HTTP 404" in check.detail

    @pytest.mark.parametrize(
        "payload",
        [{"results": []}, {}, {"results": [{"content": ""}]}, {"results": ["junk"]}],
    )
    def test_empty(self, payload):
        assert evaluate_extract(payload).status == "FAIL"

    def test_envelope_failure(self):
        check = evaluate_extract({"success": False, "error": "no provider"})
        assert (check.status, check.detail) == ("FAIL", "error: no provider")


# ---------------------------------------------------------------------------
# Expected-error evaluators and notice
# ---------------------------------------------------------------------------


class TestEvaluateErrors:
    def test_search_error_pass(self):
        payload = {"success": False, "error": "DEEPSEEK_API_KEY is not set"}
        assert evaluate_search_error(payload, "DEEPSEEK_API_KEY is not set", "n").status == "PASS"

    def test_search_error_rescued_fails(self):
        check = evaluate_search_error(_RESCUED, "DEEPSEEK_API_KEY is not set", "n")
        assert check.status == "FAIL"
        assert check.detail.startswith("expected a deepseek error but served by 'exa'")

    def test_search_error_wrong_message(self):
        check = evaluate_search_error({"success": False, "error": "other"}, "HTTP 401", "n")
        assert check.status == "FAIL"
        assert "'HTTP 401'" in check.detail

    def test_search_error_unexpected_success(self):
        assert evaluate_search_error({"success": True, "data": {"web": []}}, "HTTP 401", "n").status == "FAIL"

    def test_extract_error_pass_from_entry(self):
        payload = {"results": [{"url": "u", "title": "", "content": "", "error": "DEEPSEEK_API_KEY is not set"}]}
        assert evaluate_extract_error(payload, "DEEPSEEK_API_KEY is not set", "n").status == "PASS"

    def test_extract_error_pass_from_envelope(self):
        payload = {"success": False, "error": "DEEPSEEK_API_KEY is not set"}
        assert evaluate_extract_error(payload, "DEEPSEEK_API_KEY is not set", "n").status == "PASS"

    def test_extract_error_rescued_content_fails(self):
        payload = {"results": [{"url": "u", "title": "Example Domain", "content": "# Example", "error": None}]}
        assert evaluate_extract_error(payload, "DEEPSEEK_API_KEY is not set", "n").status == "FAIL"


class TestEvaluateNotice:
    def test_present(self):
        payload = {"success": True, "data": {"web": []}, "notice": "Query was ... shortened to: 'x'"}
        assert evaluate_notice(payload).status == "PASS"

    def test_absent(self):
        assert evaluate_notice({"success": True, "data": {"web": []}}).status == "FAIL"

    def test_rescued(self):
        assert evaluate_notice({**_RESCUED, "notice": "x"}).status == "FAIL"


# ---------------------------------------------------------------------------
# Summary, formatting, config, CLI
# ---------------------------------------------------------------------------


class TestSummaryAndFormatting:
    def test_summarize_all_good(self):
        assert summarize([Check("a", "PASS"), Check("b", "SKIP")]) == ("1 passed, 0 failed, 1 skipped", 0)

    def test_summarize_with_failure(self):
        assert summarize([Check("a", "PASS"), Check("b", "FAIL")]) == ("1 passed, 1 failed, 0 skipped", 1)

    def test_line_with_and_without_detail(self):
        assert Check("n", "PASS", "d").line() == "[PASS] n — d"
        assert Check("n", "SKIP").line() == "[SKIP] n"

    def test_long_query_exceeds_limit(self):
        assert len(smoke._LONG_QUERY) > 128
        assert " " in smoke._LONG_QUERY


class TestIsolatedConfig:
    def test_selects_deepseek_enables_plugin_disables_rescue(self):
        lines = isolated_config_text().splitlines()
        assert "web:" in lines
        assert "  backend: deepseek" in lines
        assert "  keyless_fallback: false" in lines
        assert "  keyless_rescue: false" in lines
        assert "plugins:" in lines
        assert "    - web-deepseek" in lines


class TestCli:
    def test_defaults(self):
        args = parse_args([])
        assert (args.current_home, args.register, args.skip_live, args.quick, args.keep_home) == (
            False,
            False,
            False,
            False,
            False,
        )
        assert args.query == "DeepSeek API release notes"
        assert args.url == "https://example.com"

    def test_all_flags(self):
        args = parse_args(
            [
                "--current-home",
                "--register",
                "--skip-live",
                "--quick",
                "--keep-home",
                "--query",
                "q",
                "--url",
                "https://u",
            ]
        )
        assert (args.current_home, args.register, args.skip_live, args.quick, args.keep_home) == (
            True,
            True,
            True,
            True,
            True,
        )
        assert (args.query, args.url) == ("q", "https://u")

    def test_main_with_hermes_delegates_to_report(self, monkeypatch):
        for name in ("tools", "tools.web_tools", "agent", "agent.web_search_registry"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        seen = []
        monkeypatch.setattr(smoke, "_report", lambda args: seen.append(args) or 0)
        assert smoke.main(["--skip-live"]) == 0
        assert seen[0].skip_live is True

    def test_main_without_hermes_exits_2(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "tools.web_tools", None)  # force ImportError regardless of test order
        assert smoke.main([]) == 2
        assert "Hermes Agent is not importable" in capsys.readouterr().out


class TestPluginDirToMirror:
    @staticmethod
    def _entry_points(names):
        eps = [types.SimpleNamespace(name=n) for n in names]
        return lambda group: eps if group == "hermes_agent.plugins" else []

    def test_none_when_entry_point_installed(self, monkeypatch):
        monkeypatch.setattr(smoke.importlib.metadata, "entry_points", self._entry_points(["web-deepseek"]))
        assert smoke.plugin_dir_to_mirror() is None

    def test_checkout_root_when_only_directory_install(self, monkeypatch):
        monkeypatch.setattr(smoke.importlib.metadata, "entry_points", self._entry_points(["other-plugin"]))
        root = smoke.plugin_dir_to_mirror()
        assert root is not None
        assert (root / "plugin.yaml").is_file()
        assert (root / "src" / "hermes_deepseek_search" / "smoke.py").is_file()

    def test_none_outside_a_plugin_checkout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(smoke.importlib.metadata, "entry_points", self._entry_points([]))
        monkeypatch.setattr(smoke, "__file__", str(tmp_path / "site-packages" / "hermes_deepseek_search" / "smoke.py"))
        assert smoke.plugin_dir_to_mirror() is None
