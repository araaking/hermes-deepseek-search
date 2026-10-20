"""``hermes web-deepseek smoke [flags]``: run the smoke test with the interpreter that runs Hermes.

Hermes 0.22+ manages its own Python environment, so there is no fixed venv path to type. A plugin CLI command
runs inside Hermes, where ``sys.executable`` is that interpreter. The smoke test still runs in a child process:
isolated mode must set ``HERMES_HOME`` before Hermes is imported.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import hermes_deepseek_search

COMMAND = "web-deepseek"
HELP = "DeepSeek web provider tools (smoke test)"


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["smoke"], help="smoke: end-to-end check of web_search / web_extract")
    parser.add_argument(
        "smoke_args",
        nargs=argparse.REMAINDER,
        help="flags for the smoke test, e.g. --current-home, --skip-live (see 'smoke --help')",
    )


def smoke_command(smoke_args: list[str]) -> tuple[list[str], dict[str, str]]:
    """Command line and environment for the child process. ``PYTHONPATH`` leads with the directory this package
    was imported from, so the child runs the same copy even when it is not installed (a directory-plugin
    fallback)."""
    package_root = str(Path(hermes_deepseek_search.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH", "")) if p)
    return [sys.executable, "-m", "hermes_deepseek_search.smoke", *smoke_args], env


def handle(args: argparse.Namespace) -> int:
    cmd, env = smoke_command(list(args.smoke_args))
    return subprocess.run(cmd, env=env, check=False).returncode
