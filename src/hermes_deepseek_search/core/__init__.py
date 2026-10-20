"""Hermes-free DeepSeek Search & Fetch client — the shared core for every adapter."""

from __future__ import annotations

from hermes_deepseek_search.core.client import API_KEY_ENV, DEFAULT_BASE_URL, DeepSeekClient
from hermes_deepseek_search.core.errors import (
    AuthenticationError,
    ContentNotFoundError,
    DeepSeekAPIError,
    DeepSeekError,
    FetchBlockedError,
    GatewayTimeoutError,
    InsufficientBalanceError,
    InvalidRequestError,
    MissingApiKeyError,
    NetworkError,
    PayloadTooLargeError,
    RateLimitError,
    RequestTimeoutError,
    ResponseParseError,
    ServerError,
    ValidationError,
)
from hermes_deepseek_search.core.models import FetchResult, SearchResponse, SearchResult

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_BASE_URL",
    "AuthenticationError",
    "ContentNotFoundError",
    "DeepSeekAPIError",
    "DeepSeekClient",
    "DeepSeekError",
    "FetchBlockedError",
    "FetchResult",
    "GatewayTimeoutError",
    "InsufficientBalanceError",
    "InvalidRequestError",
    "MissingApiKeyError",
    "NetworkError",
    "PayloadTooLargeError",
    "RateLimitError",
    "RequestTimeoutError",
    "ResponseParseError",
    "SearchResponse",
    "SearchResult",
    "ServerError",
    "ValidationError",
]
