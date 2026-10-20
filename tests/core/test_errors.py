"""Tests for the typed error hierarchy and HTTP → exception mapping.

Covers:
- error_from_response() — class per status, retryable flags, fallback classes for unmapped codes
- Message format — label, status, server message, hint; empty / non-JSON / non-object bodies
- parse_retry_after() — numeric, fractional, garbage, negative, missing
- ValidationError doubles as ValueError
"""

from __future__ import annotations

import httpx
import pytest

from hermes_deepseek_search.core import (
    AuthenticationError,
    ContentNotFoundError,
    DeepSeekAPIError,
    FetchBlockedError,
    GatewayTimeoutError,
    InsufficientBalanceError,
    InvalidRequestError,
    PayloadTooLargeError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    ValidationError,
)
from hermes_deepseek_search.core.errors import error_from_response, parse_retry_after

_LABEL = "DeepSeek Search"


def _err(status: int, body: str = "", headers: dict[str, str] | None = None) -> DeepSeekAPIError:
    return error_from_response(status, body, httpx.Headers(headers or {}), label=_LABEL)


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("status", "cls", "retryable"),
        [
            (400, InvalidRequestError, False),
            (401, AuthenticationError, False),
            (402, InsufficientBalanceError, False),
            (403, FetchBlockedError, False),
            (404, ContentNotFoundError, False),
            (408, RequestTimeoutError, True),
            (413, PayloadTooLargeError, False),
            (429, RateLimitError, True),
            (500, ServerError, True),
            (502, ServerError, True),
            (503, ServerError, True),
            (504, GatewayTimeoutError, True),
        ],
    )
    def test_maps_status_to_class(self, status, cls, retryable):
        err = _err(status)
        assert type(err) is cls
        assert err.retryable is retryable
        assert err.status_code == status

    def test_unmapped_4xx_is_base_api_error(self):
        err = _err(418)
        assert type(err) is DeepSeekAPIError
        assert err.retryable is False

    def test_unmapped_5xx_is_server_error(self):
        err = _err(599)
        assert type(err) is ServerError
        assert err.retryable is True

    def test_gateway_timeout_is_server_error(self):
        assert isinstance(_err(504), ServerError)


# ---------------------------------------------------------------------------
# Message format
# ---------------------------------------------------------------------------


class TestErrorMessage:
    def test_includes_label_status_server_message_and_hint(self):
        body = (
            '{"error": {"message": "bad key", "type": "invalid_request_error", '
            '"param": null, "code": "invalid_request_error"}}'
        )
        err = _err(401, body)
        assert str(err) == (
            "DeepSeek Search returned HTTP 401: bad key. "
            "Check that DEEPSEEK_API_KEY is set to a valid DeepSeek API key."
        )
        assert err.code == "invalid_request_error"

    def test_empty_body_has_no_detail(self):
        assert str(_err(500, "")) == "DeepSeek Search returned HTTP 500"

    def test_non_json_body_truncated(self):
        err = _err(502, "<html>" + "x" * 1000 + "</html>")
        detail = str(err).removeprefix("DeepSeek Search returned HTTP 502: ")
        assert len(detail) == 300

    def test_json_list_body_does_not_crash(self):
        err = _err(400, "[1,2]")
        assert "[1,2]" in str(err)
        assert err.code is None

    def test_error_without_message_falls_back_to_body(self):
        err = _err(400, '{"error": {}}')
        assert '{"error": {}}' in str(err)

    def test_hint_appended_for_balance(self):
        assert "top up" in str(_err(402, '{"error": {"message": "Insufficient Balance"}}'))


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------


class TestRetryAfter:
    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"Retry-After": "2"}, 2.0),
            ({"Retry-After": "1.5"}, 1.5),
            ({"Retry-After": "abc"}, None),
            ({"Retry-After": "-1"}, None),
            ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
            ({}, None),
        ],
    )
    def test_parse_retry_after(self, headers, expected):
        assert parse_retry_after(httpx.Headers(headers)) == expected

    def test_rate_limit_error_carries_retry_after(self):
        err = _err(429, '{"error": {"message": "slow down"}}', {"Retry-After": "3"})
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 3.0


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_validation_error_is_value_error(self):
        assert issubclass(ValidationError, ValueError)
