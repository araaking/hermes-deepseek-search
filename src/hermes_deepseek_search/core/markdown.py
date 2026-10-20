"""Strip Markdown backslash escapes from Fetch API page text.

DeepSeek's HTML-to-Markdown conversion escapes literal punctuation (``\\#``, ``\\-``, ``\\*`` …) so that a
renderer would not read it as syntax. The text goes to a model, not a renderer, so the escapes are only noise.
Fenced code blocks and inline code spans are left untouched, because a backslash there is literal (regexes,
Windows paths). Indented code blocks cannot be told apart from indented list text and are unescaped like prose.
"""

from __future__ import annotations

import re

# CommonMark: a backslash before any ASCII punctuation character is an escape.
_PUNCTUATION = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
_BACKTICKS = re.compile(r"`+")


def unescape_markdown(text: str) -> str:
    if "\\" not in text:
        return text
    out: list[str] = []
    fence = ""  # the opening fence while inside a fenced code block
    for line in text.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence:
            if match and _closes(match.group(1), fence) and not line[match.end() :].strip():
                fence = ""
            out.append(line)
        elif match:
            fence = match.group(1)
            out.append(line)
        else:
            out.append(_unescape_line(line))
    return "".join(out)


def _closes(candidate: str, opening: str) -> bool:
    return candidate[0] == opening[0] and len(candidate) >= len(opening)


def _unescape_line(line: str) -> str:
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and i + 1 < n and line[i + 1] in _PUNCTUATION:
            out.append(line[i + 1])
            i += 2
        elif ch == "`":
            run = _BACKTICKS.match(line, i).end() - i
            end = _code_span_end(line, i + run, run)
            out.append(line[i:end])
            i = end
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _code_span_end(line: str, start: int, run: int) -> int:
    """Index just past the backtick run of exactly ``run`` that closes a code span opened before ``start``; when
    there is none, just past the opening run, which is then literal text."""
    for match in _BACKTICKS.finditer(line, start):
        if len(match.group()) == run:
            return match.end()
    return start
