"""Typed errors for the DeepSeek Search & Fetch API.

``str(exc)`` is always a complete, model-readable sentence — adapters surface it verbatim
as ``{"success": False, "error": str(exc)}``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_MAX_DETAIL_CHARS = 300


class DeepSeekError(Exception):
    """Base class. ``retryable`` drives the client's retry loop."""

    retryable: bool = False

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class ValidationError(DeepSeekError, ValueError):
    """Rejected locally before any request was sent (no cost)."""


class MissingApiKeyError(DeepSeekError):
    """No API key configured."""


class NetworkError(DeepSeekError):
    """Transport failure (DNS, connect, read timeout). Safe to retry: the API is read-only."""

    retryable = True


class ResponseParseError(DeepSeekError):
    """HTTP 200 whose body is not the JSON shape we expect."""


class DeepSeekAPIError(DeepSeekError):
    """Non-200 reply from the API. Subclasses add a ``hint`` telling the user what to do; ``retry_after`` holds
    the server's numeric ``Retry-After`` seconds for retryable statuses (None otherwise)."""

    hint: str = ""

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InvalidRequestError(DeepSeekAPIError):
    """HTTP 400."""

    hint = "The request was rejected as invalid; change the query or URL before retrying."


class AuthenticationError(DeepSeekAPIError):
    """HTTP 401."""

    hint = "Check that DEEPSEEK_API_KEY is set to a valid DeepSeek API key."


class InsufficientBalanceError(DeepSeekAPIError):
    """HTTP 402."""

    hint = "The DeepSeek account balance is insufficient; top up at https://platform.deepseek.com."


class FetchBlockedError(DeepSeekAPIError):
    """HTTP 403."""

    hint = "DeepSeek's security policy refused to fetch this URL."


class ContentNotFoundError(DeepSeekAPIError):
    """HTTP 404."""

    hint = "No usable page content could be retrieved (unavailable URL, site policy, or unsupported content type)."


class RequestTimeoutError(DeepSeekAPIError):
    """HTTP 408."""

    retryable = True


class PayloadTooLargeError(DeepSeekAPIError):
    """HTTP 413."""

    hint = "The request body is too large."


class RateLimitError(DeepSeekAPIError):
    """HTTP 429."""

    retryable = True
    hint = "DeepSeek QPS limit exceeded (Search 100/s, Fetch 25/s per account, plus a per-domain Fetch limit)."


class ServerError(DeepSeekAPIError):
    """HTTP 5xx."""

    retryable = True


class GatewayTimeoutError(ServerError):
    """HTTP 504."""

    hint = "A downstream request timed out."


_STATUS_MAP: dict[int, type[DeepSeekAPIError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    402: InsufficientBalanceError,
    403: FetchBlockedError,
    404: ContentNotFoundError,
    408: RequestTimeoutError,
    413: PayloadTooLargeError,
    429: RateLimitError,
    504: GatewayTimeoutError,
}


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Numeric ``Retry-After`` seconds, or None (HTTP-date form and negatives are ignored)."""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _server_message(body: str) -> tuple[str, str | None]:
    """``(message, code)`` from an OpenAI-style error body; falls back to the (trimmed) raw text."""
    try:
        err = json.loads(body).get("error") or {}
        message = str(err.get("message") or "").strip()
        code = err.get("code")
    except (ValueError, AttributeError):
        message, code = "", None
    return (message or body.strip())[:_MAX_DETAIL_CHARS], code


def error_from_response(status_code: int, body: str, headers: Mapping[str, str], *, label: str) -> DeepSeekAPIError:
    """Map a non-200 reply to its typed error. ``label`` is e.g. ``"DeepSeek Search"``."""
    cls = _STATUS_MAP.get(status_code) or (ServerError if status_code >= 500 else DeepSeekAPIError)
    detail, code = _server_message(body)
    message = f"{label} returned HTTP {status_code}" + (f": {detail}" if detail else "")
    if cls.hint:
        message = f"{message}. {cls.hint}"
    retry_after = parse_retry_after(headers) if cls.retryable else None
    return cls(message, status_code=status_code, code=code, retry_after=retry_after)
