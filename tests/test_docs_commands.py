"""The documentation must keep describing this repository.

Runs the static documentation check as part of the normal test suite, so a
rename or a deleted file breaks the build instead of silently rotting a guide.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_docs_commands", ROOT / "tools" / "check_docs_commands.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_docs_commands"] = module
    spec.loader.exec_module(module)
    return module


def test_documentation_links_paths_and_commands_resolve(capsys):
    checker = _load_checker()
    exit_code = checker.main()
    captured = capsys.readouterr().out
    assert exit_code == 0, "documentation check failed:\n" + captured


def test_every_documented_guide_exists():
    for name in ("getting-started", "controllers", "configuration",
                 "experiments", "reproducing-the-paper"):
        assert (ROOT / "docs" / (name + ".md")).is_file(), name


def test_readme_links_the_documentation_index():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("getting-started", "controllers", "configuration",
                 "experiments", "reproducing-the-paper"):
        assert "docs/{}.md".format(name) in readme, name
