"""Directory-plugin entry for ``hermes plugins install araaking/hermes-deepseek-search`` (Hermes 0.22+).

Hermes clones this repository to ``~/.hermes/plugins/web-deepseek`` and builds ``pyproject.toml`` into its
managed environment, so the real package is normally importable and this file only forwards to it. When it is
not (a hand-copied directory, or ``--no-deps``), the bundled ``src/`` tree is used instead. The import happens
inside ``register`` so that importing this file has no side effects (pytest imports it as the rootdir package).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

__all__ = ["register"]


def register(ctx: Any) -> None:
    try:
        from hermes_deepseek_search.hermes import register as register_package
    except ModuleNotFoundError as exc:
        if exc.name != "hermes_deepseek_search":
            raise
        src = str(Path(__file__).resolve().parent / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from hermes_deepseek_search.hermes import register as register_package
    register_package(ctx)
