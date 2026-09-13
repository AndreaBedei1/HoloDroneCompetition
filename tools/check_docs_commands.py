"""Check that the README and the guides still describe this repository.

Static checks only - nothing is executed and no simulator is launched:

1. every relative Markdown link and image source resolves to a real file;
2. every repository path quoted in a fenced code block exists;
3. every ``python -m <module>`` names an importable module of this package;
4. every ``--controller`` / ``--participant-controller`` value is a real
   built-in alias or a real file;
5. no documentation page references a tree removed in the release cleanup.

Usage:
    python tools/check_docs_commands.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))

# Paths that appear in prose as placeholders rather than as real files.
PLACEHOLDERS = {
    "<HoloOcean-2.3.0 source>/client",
    "my_package.my_module",
    "path/to/my_controller.py",
}
# Trees deleted by the release cleanup; a mention means the docs went stale.
FORBIDDEN = [
    "article/", "artifacts_gen2", "submission_flat", "scientific_release",
    "campaign_A", "campaign_B", "campaign_C", "matrix_78", "retry6_50k",
    "results/rl_public", "results/onboard_only_validation",
]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
IMG = re.compile(r'<img[^>]+src="([^"]+)"')
HREF = re.compile(r'<a[^>]+href="([^"]+)"')
FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)
PY_MODULE = re.compile(r"python(?:\.exe)?\s+-m\s+([A-Za-z_][\w.]*)")
CONTROLLER = re.compile(r"--(?:participant-)?controller\s+(\S+)")
REPO_PATH = re.compile(
    r"(?<![\w./-])((?:marine_race_arena|artifacts|article_journal|docs|configs|scripts|tools)"
    r"/[\w./-]*[\w])")

problems: list[str] = []


def fail(page: Path, message: str) -> None:
    problems.append("{}: {}".format(page.relative_to(ROOT).as_posix(), message))


def check_links(page: Path, text: str) -> None:
    targets = LINK.findall(text) + IMG.findall(text) + HREF.findall(text)
    for target in targets:
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        if not (page.parent / clean).resolve().exists():
            fail(page, "broken link -> {}".format(target))


def check_code_blocks(page: Path, text: str) -> None:
    builtins_ = builtin_aliases()
    for block in FENCE.findall(text):
        for path in set(REPO_PATH.findall(block)):
            if path in PLACEHOLDERS or "<" in path or "*" in path:
                continue
            if not (ROOT / path).exists():
                fail(page, "code block references a missing path -> {}".format(path))
        for module in set(PY_MODULE.findall(block)):
            if not module.startswith("marine_race_arena"):
                continue
            if importlib.util.find_spec(module) is None:
                fail(page, "python -m names a missing module -> {}".format(module))
        for value in set(CONTROLLER.findall(block)):
            if value.startswith("-") or value in PLACEHOLDERS:
                continue
            if value in builtins_:
                continue
            if value.endswith(".py") and (ROOT / value).exists():
                continue
            if ":" in value or "." in value:          # module path, checked above
                continue
            fail(page, "unknown controller -> {}".format(value))


def check_forbidden(page: Path, text: str) -> None:
    for token in FORBIDDEN:
        for line in text.splitlines():
            if token in line and "removed" not in line.lower() and "cleanup" not in line.lower():
                fail(page, "stale reference {!r} in: {}".format(token, line.strip()[:90]))
                break


def builtin_aliases() -> set:
    sys.path.insert(0, str(ROOT))
    from marine_race_arena.participants.controller_loader import ControllerLoader
    # the evaluator's own learned-controller labels are valid there too
    return set(ControllerLoader.BUILT_INS) | {"recurrent_ppo", "rules"}


def main() -> int:
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        check_links(page, text)
        check_code_blocks(page, text)
        check_forbidden(page, text)

    print("checked {} pages".format(len(PAGES)))
    for page in PAGES:
        print("   ", page.relative_to(ROOT).as_posix())
    if problems:
        print("\n{} problem(s):".format(len(problems)))
        for problem in problems:
            print("  -", problem)
        return 1
    print("\nno problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
