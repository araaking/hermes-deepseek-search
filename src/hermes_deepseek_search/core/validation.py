"""Local request validation mirroring the DeepSeek API limits (rejects before spending a request)."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from hermes_deepseek_search.core.errors import ValidationError

MAX_QUERY_CHARS = 128
MAX_ORI_QUESTION_CHARS = 128
MIN_COUNT = 1
MAX_COUNT = 50
MAX_URL_CHARS = 8192
_URL_PREVIEW_CHARS = 200
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_query(query: str) -> tuple[str, bool]:
    """Collapse whitespace and fit ``query`` into 128 Unicode chars → ``(query, truncated)``.

    Over-long queries are cut at the last space when that keeps at least half the budget
    (so words/operators are not split), else hard-cut. Empty/whitespace-only raises.
    """
    q = " ".join((query or "").split())
    if not q:
        raise ValidationError("Search query must not be empty or whitespace only.")
    if len(q) <= MAX_QUERY_CHARS:
        return q, False
    cut = q[:MAX_QUERY_CHARS]
    space = cut.rfind(" ")
    if space >= MAX_QUERY_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip(), True


def clamp_count(count: int) -> int:
    """Coerce to int and clamp into 1..50."""
    try:
        n = int(count)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"count must be an integer between {MIN_COUNT} and {MAX_COUNT}.") from exc
    return max(MIN_COUNT, min(n, MAX_COUNT))


def validate_ori_question(ori_question: str | None) -> str | None:
    """Optional hint: blank → None, over-long → hard-cut to 128 chars (never rejected)."""
    if ori_question is None:
        return None
    q = " ".join(ori_question.split())
    return q[:MAX_ORI_QUESTION_CHARS] or None


def validate_localization(localization: str | None) -> str | None:
    """Optional IP address used to localize results; blank → None, invalid → ValidationError."""
    value = (localization or "").strip()
    if not value:
        return None
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValidationError(f"localization must be an IP address, got {value!r}.") from exc
    return value


def validate_url(url: str) -> str:
    """http(s) URL with a host, no userinfo, ≤ 8192 chars → stripped URL."""
    u = (url or "").strip()
    if not u:
        raise ValidationError("URL must not be empty.")
    preview = u[:_URL_PREVIEW_CHARS]
    if len(u) > MAX_URL_CHARS:
        raise ValidationError(f"URL is longer than {MAX_URL_CHARS} characters.")
    if any(c.isspace() or not c.isprintable() for c in u):
        raise ValidationError(f"URL must not contain whitespace or control characters: {preview!r}.")
    try:
        parts = urlsplit(u)
        host = parts.hostname
        has_userinfo = parts.username is not None or parts.password is not None
    except ValueError as exc:  # e.g. unterminated IPv6 literal
        raise ValidationError(f"URL is malformed: {preview!r}.") from exc
    if parts.scheme.lower() not in ("http", "https"):
        raise ValidationError(f"Only http/https URLs can be fetched, got {preview!r}.")
    if not host:
        raise ValidationError(f"URL has no host: {preview!r}.")
    if has_userinfo:
        raise ValidationError("URL must not contain a username or password.")
    return u


def validate_base_url(base_url: str) -> str:
    """API endpoint that receives the Bearer key: https (plain http only for loopback), a host, no userinfo,
    no whitespace → URL without trailing slash. Messages never echo the URL (it may embed credentials)."""
    u = (base_url or "").strip().rstrip("/")
    try:
        parts = urlsplit(u)
        host = (parts.hostname or "").lower()
        has_userinfo = parts.username is not None or parts.password is not None
    except ValueError as exc:
        raise ValidationError("base_url is not a valid URL.") from exc
    scheme = parts.scheme.lower()
    if not host or scheme not in ("https", "http") or any(c.isspace() or not c.isprintable() for c in u):
        raise ValidationError("base_url must be an https:// URL with a host.")
    if scheme == "http" and host not in _LOOPBACK_HOSTS:
        raise ValidationError(
            "base_url must use https:// (plain http is only allowed for localhost); it carries the API key."
        )
    if has_userinfo:
        raise ValidationError("base_url must not contain a username or password.")
    return u
