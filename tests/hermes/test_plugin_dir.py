"""Tests for the Hermes 0.22+ directory-plugin route and the ``hermes web-deepseek smoke`` command.

Covers:
- Root ``__init__.py`` — loaded the way Hermes loads a directory plugin, it forwards ``register(ctx)`` to the
  package; falls back to the bundled ``src/`` when the package is not importable; re-raises other import errors
- ``plugin.yaml`` — id equals the pip entry point (so Hermes treats both routes as one plugin), version in sync
- CLI — argument parsing, child-process command and ``PYTHONPATH``, exit code passthrough
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import hermes_deepseek_search
from hermes_deepseek_search.hermes import cli
from hermes_deepseek_search.hermes.provider import DeepSeekWebSearchProvider
from tests.hermes.conftest import FakeContext

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = "hermes_deepseek_search"


def _load_wrapper():
    """Load the repo-root ``__init__.py`` as Hermes does (``hermes_plugins.<slug>`` with the plugin dir as the
    package path)."""
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins_test.web_deepseek", _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _HideOnce(importlib.abc.MetaPathFinder):
    """Makes the first import of ``hermes_deepseek_search`` fail with *missing* as the missing module name."""

    def __init__(self, missing: str = _PACKAGE) -> None:
        self.missing = missing
        self.hits = 0

    def find_spec(self, fullname, path, target=None):
        if fullname == _PACKAGE and self.hits == 0:
            self.hits += 1
            raise ModuleNotFoundError(f"No module named {self.missing!r}", name=self.missing)
        return None


@pytest.fixture
def unimported_package(monkeypatch):
    """Forget every imported ``hermes_deepseek_search`` module (restored afterwards) and snapshot ``sys.path``."""
    for name in [m for m in sys.modules if m == _PACKAGE or m.startswith(f"{_PACKAGE}.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


class TestWrapper:
    def test_import_has_no_side_effects(self, unimported_package):
        module = _load_wrapper()
        assert callable(module.register)
        assert _PACKAGE not in sys.modules

    def test_register_forwards_to_package(self):
        ctx = FakeContext()
        _load_wrapper().register(ctx)
        assert len(ctx.providers) == 1
        assert ctx.providers[0].name == "deepseek"

    def test_falls_back_to_bundled_src(self, unimported_package):
        src = _ROOT / "src"
        sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != src]  # drop the editable-install entry
        hider = _HideOnce()
        sys.meta_path.insert(0, hider)
        ctx = FakeContext()
        _load_wrapper().register(ctx)
        assert hider.hits == 1
        assert sys.path[0] == str(_ROOT / "src")
        assert type(ctx.providers[0]).__name__ == DeepSeekWebSearchProvider.__name__

    def test_other_missing_module_is_reraised(self, unimported_package):
        sys.meta_path.insert(0, _HideOnce(missing="httpx"))
        with pytest.raises(ModuleNotFoundError) as info:
            _load_wrapper().register(FakeContext())
        assert info.value.name == "httpx"


class TestManifest:
    @staticmethod
    def _field(name: str) -> str:
        text = (_ROOT / "plugin.yaml").read_text(encoding="utf-8")
        match = re.search(rf"^{name}:\s*(.+?)\s*$", text, re.M)
        assert match, f"plugin.yaml has no top-level {name!r}"
        return match.group(1)

    def test_id_matches_entry_point(self):
        eps = importlib.metadata.entry_points(group="hermes_agent.plugins")
        assert self._field("name") in {e.name for e in eps if e.value == "hermes_deepseek_search.hermes"}

    def test_versions_in_sync(self):
        project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert self._field("version") == project["version"] == hermes_deepseek_search.__version__

    def test_required_fields_present(self):
        for name in ("name", "version", "description"):
            assert self._field(name)


class TestCli:
    @staticmethod
    def _parse(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="hermes web-deepseek")
        cli.setup_parser(parser)
        return parser.parse_args(argv)

    def test_parses_smoke_flags(self):
        args = self._parse(["smoke", "--current-home", "--skip-live", "--query", "x y"])
        assert args.action == "smoke"
        assert args.smoke_args == ["--current-home", "--skip-live", "--query", "x y"]

    def test_bare_smoke(self):
        assert self._parse(["smoke"]).smoke_args == []

    def test_unknown_action_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["nope"])

    def test_child_command_uses_current_interpreter_and_package_copy(self, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "existing")
        cmd, env = cli.smoke_command(["--skip-live"])
        assert cmd == [sys.executable, "-m", "hermes_deepseek_search.smoke", "--skip-live"]
        package_root = str(Path(hermes_deepseek_search.__file__).resolve().parent.parent)
        assert env["PYTHONPATH"] == os.pathsep.join([package_root, "existing"])

    def test_child_pythonpath_without_existing_value(self, monkeypatch):
        monkeypatch.delenv("PYTHONPATH", raising=False)
        _, env = cli.smoke_command([])
        assert env["PYTHONPATH"] == str(Path(hermes_deepseek_search.__file__).resolve().parent.parent)

    def test_handle_returns_child_exit_code(self, monkeypatch):
        seen = {}

        def fake_run(cmd, env, check):
            seen.update(cmd=cmd, check=check)
            return subprocess.CompletedProcess(cmd, 1)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        assert cli.handle(argparse.Namespace(action="smoke", smoke_args=["--quick"])) == 1
        assert seen == {"cmd": [sys.executable, "-m", "hermes_deepseek_search.smoke", "--quick"], "check": False}
