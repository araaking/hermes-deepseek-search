"""Tests for unescape_markdown: prose escapes removed, code left literal, everything else untouched."""

from __future__ import annotations

import pytest

from hermes_deepseek_search.core import unescape_markdown


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"\# Treatment price list", "# Treatment price list"),
        (r"\- 60 minutes \- IDR 1.500.000", "- 60 minutes - IDR 1.500.000"),
        (r"1\. Consultation", "1. Consultation"),
        (r"\*\*Promo\*\* \_today\_", "**Promo** _today_"),
        (r"\[Book now\]\(https://x.example\)", "[Book now](https://x.example)"),
        (r"5 \+ 3 \= 8 \> 7 \< 9 \| \~ \! \{ \}", "5 + 3 = 8 > 7 < 9 | ~ ! { }"),
        (r"a \\ b", "a \\ b"),  # escaped backslash
        (r"\`not code\`", "`not code`"),  # escaped backticks open no code span
    ],
)
def test_prose_escapes_removed(raw, expected):
    assert unescape_markdown(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain text, no backslash",
        r"C:\Users\clinic",  # backslash before a letter is not an escape
        "line one\\\nline two",  # trailing backslash = hard line break
        r"price \5",
        "\\",
    ],
)
def test_non_escapes_untouched(text):
    assert unescape_markdown(text) == text


def test_inline_code_kept_literal():
    assert unescape_markdown(r"run `grep \-r x` then \- done") == r"run `grep \-r x` then - done"


def test_double_backtick_code_span():
    assert unescape_markdown(r"``a ` \# b`` \#") == r"``a ` \# b`` #"


def test_unclosed_backtick_is_literal():
    assert unescape_markdown(r"a ` b \# c") == "a ` b # c"


def test_fenced_code_block_kept_literal():
    raw = "\\# Title\n```python\nre.compile(r'\\d+\\-\\d+')\n```\n\\- after\n"
    assert unescape_markdown(raw) == "# Title\n```python\nre.compile(r'\\d+\\-\\d+')\n```\n- after\n"


def test_tilde_fence_and_longer_closing_fence():
    raw = "~~~~\n\\#\n~~~\n\\#\n~~~~~\n\\# out\n"
    # `~~~` is shorter than the opening `~~~~`, so it does not close the block
    assert unescape_markdown(raw) == "~~~~\n\\#\n~~~\n\\#\n~~~~~\n# out\n"


def test_fence_with_trailing_text_does_not_close():
    raw = "```\n\\#\n``` not a close\n\\#\n```\n\\#"
    assert unescape_markdown(raw) == "```\n\\#\n``` not a close\n\\#\n```\n#"


def test_unterminated_fence_keeps_rest_literal():
    assert unescape_markdown("```\n\\# a\n\\# b") == "```\n\\# a\n\\# b"


def test_line_endings_preserved():
    assert unescape_markdown("\\# a\r\n\\- b\r\n") == "# a\r\n- b\r\n"
