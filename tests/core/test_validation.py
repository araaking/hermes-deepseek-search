"""Tests for local request validation.

Covers:
- validate_query() — whitespace collapse, empty rejection, 128-char boundary, word-boundary and hard truncation, Unicode
- clamp_count() — clamping into 1..50, numeric strings, non-numeric rejection
- validate_ori_question() — None/blank passthrough, hard truncation
- validate_localization() — IPv4/IPv6 accepted, blank → None, garbage rejected
- validate_url() — scheme, host, userinfo, length, malformed IPv6
"""

from __future__ import annotations

import pytest

from hermes_deepseek_search.core import ValidationError
from hermes_deepseek_search.core.validation import (
    MAX_QUERY_CHARS,
    clamp_count,
    validate_base_url,
    validate_localization,
    validate_ori_question,
    validate_query,
    validate_url,
)

# ---------------------------------------------------------------------------
# validate_query
# ---------------------------------------------------------------------------


class TestValidateQuery:
    def test_collapses_whitespace(self):
        assert validate_query("  DeepSeek  V4\n release ") == ("DeepSeek V4 release", False)

    @pytest.mark.parametrize("query", ["", "   ", "\n\t", None])
    def test_empty_or_whitespace_rejected(self, query):
        with pytest.raises(ValidationError):
            validate_query(query)

    def test_exactly_128_chars_kept(self):
        q = "a" * MAX_QUERY_CHARS
        assert validate_query(q) == (q, False)

    def test_long_query_truncated_at_word_boundary(self):
        q = " ".join(["word"] * 50)  # 249 chars
        out, truncated = validate_query(q)
        assert truncated is True
        assert len(out) <= MAX_QUERY_CHARS
        assert out.endswith("word")
        assert not out.endswith(" ")

    def test_long_query_without_spaces_hard_cut(self):
        out, truncated = validate_query("a" * 200)
        assert truncated is True
        assert out == "a" * MAX_QUERY_CHARS

    def test_space_too_early_falls_back_to_hard_cut(self):
        q = "ab " + "c" * 200
        out, truncated = validate_query(q)
        assert truncated is True
        assert len(out) == MAX_QUERY_CHARS

    def test_unicode_counted_as_characters_not_bytes(self):
        out, truncated = validate_query("深" * 130)
        assert truncated is True
        assert out == "深" * MAX_QUERY_CHARS


# ---------------------------------------------------------------------------
# clamp_count
# ---------------------------------------------------------------------------


class TestClampCount:
    @pytest.mark.parametrize(("count", "expected"), [(0, 1), (-5, 1), (1, 1), (10, 10), (50, 50), (51, 50), (999, 50)])
    def test_clamps_into_range(self, count, expected):
        assert clamp_count(count) == expected

    def test_numeric_string_coerced(self):
        assert clamp_count("7") == 7

    @pytest.mark.parametrize("count", ["abc", None, "7.5"])
    def test_non_numeric_rejected(self, count):
        with pytest.raises(ValidationError):
            clamp_count(count)


# ---------------------------------------------------------------------------
# validate_ori_question / validate_localization
# ---------------------------------------------------------------------------


class TestValidateOriQuestion:
    def test_none_passthrough(self):
        assert validate_ori_question(None) is None

    def test_blank_becomes_none(self):
        assert validate_ori_question("   ") is None

    def test_long_question_hard_cut(self):
        assert validate_ori_question("x" * 200) == "x" * 128

    def test_whitespace_collapsed(self):
        assert validate_ori_question(" why  is\nthis? ") == "why is this?"


class TestValidateLocalization:
    @pytest.mark.parametrize("ip", ["8.8.8.8", "::1", " 1.1.1.1 "])
    def test_ip_accepted(self, ip):
        assert validate_localization(ip) == ip.strip()

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_becomes_none(self, value):
        assert validate_localization(value) is None

    def test_garbage_rejected(self):
        with pytest.raises(ValidationError, match="IP address"):
            validate_localization("not-ip")


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_valid_url_stripped(self):
        assert validate_url("  https://example.com/a?b=1 ") == "https://example.com/a?b=1"

    def test_http_allowed(self):
        assert validate_url("http://example.com") == "http://example.com"

    @pytest.mark.parametrize("url", ["ftp://x.com", "javascript:alert(1)", "example.com", "file:///etc/passwd"])
    def test_wrong_scheme_rejected(self, url):
        with pytest.raises(ValidationError, match="http/https"):
            validate_url(url)

    @pytest.mark.parametrize("url", ["https://u:p@x.com", "https://u@x.com"])
    def test_userinfo_rejected(self, url):
        with pytest.raises(ValidationError, match="username or password"):
            validate_url(url)

    def test_missing_host_rejected(self):
        with pytest.raises(ValidationError, match="no host"):
            validate_url("https://")

    @pytest.mark.parametrize("url", ["", "   ", None])
    def test_empty_rejected(self, url):
        with pytest.raises(ValidationError, match="empty"):
            validate_url(url)

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError, match="8192"):
            validate_url("https://x.com/" + "a" * 8200)

    def test_malformed_ipv6_rejected_as_validation_error(self):
        with pytest.raises(ValidationError, match="malformed"):
            validate_url("http://[::1")

    def test_long_url_preview_truncated_in_message(self):
        url = "ftp://x.com/" + "a" * 5000
        with pytest.raises(ValidationError) as excinfo:
            validate_url(url)
        assert len(str(excinfo.value)) < 300

    @pytest.mark.parametrize("url", ["https://a.example/x y", "https://a.example/\tx", "https://a.example/\x00"])
    def test_whitespace_or_control_rejected(self, url):
        with pytest.raises(ValidationError, match="whitespace or control"):
            validate_url(url)


# ---------------------------------------------------------------------------
# validate_base_url
# ---------------------------------------------------------------------------


class TestValidateBaseUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://api.deepseek.com", "https://api.deepseek.com"),
            ("  https://proxy.example/v1/ ", "https://proxy.example/v1"),
            ("http://localhost:8080", "http://localhost:8080"),
            ("HTTP://127.0.0.1", "HTTP://127.0.0.1"),
            ("http://[::1]:9000", "http://[::1]:9000"),
        ],
    )
    def test_accepted(self, url, expected):
        assert validate_base_url(url) == expected

    @pytest.mark.parametrize("url", ["", "   ", "api.deepseek.com", "ftp://x.example", "https://", "https://a b.c"])
    def test_rejected_shape(self, url):
        with pytest.raises(ValidationError, match="https:// URL with a host"):
            validate_base_url(url)

    def test_plain_http_rejected_for_remote_host(self):
        with pytest.raises(ValidationError, match="only allowed for localhost"):
            validate_base_url("http://proxy.example")

    def test_credentials_rejected_without_echo(self):
        with pytest.raises(ValidationError, match="username or password") as excinfo:
            validate_base_url("https://user:hunter2@proxy.example")
        assert "hunter2" not in str(excinfo.value)

    def test_malformed_rejected(self):
        with pytest.raises(ValidationError, match="not a valid URL"):
            validate_base_url("http://[::1")
