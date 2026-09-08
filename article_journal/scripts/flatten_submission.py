"""Produce a flat LaTeX submission directory for Elsevier Editorial Manager.

Editorial Manager can require every source file at one level, with no
subdirectories. This script reads the development tree (``sections/``,
``tables/``, ``figures/``) and writes a single flat directory in which:

* every ``\input{...}`` and ``\includegraphics{...}`` path is rewritten to a
  bare basename;
* basenames are made unique by prefixing the source subdirectory when a
  collision would occur;
* the compiled ``main.bbl`` is copied so the submission builds without BibTeX;
* the class and bibliography style files are copied when they can be located.

The development tree is never modified. The output directory is rebuilt from
scratch on every run.

Usage:
    python article_journal/scripts/flatten_submission.py
    python article_journal/scripts/flatten_submission.py --out /path/to/dir
    python article_journal/scripts/flatten_submission.py --build   # test-compile
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

JOURNAL = Path(__file__).resolve().parents[1]
DEFAULT_OUT = JOURNAL / "submission_flat"

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")

GRAPHICS_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".eps")


def resolve_input(name: str) -> Path | None:
    """Resolve a \\input target relative to the manuscript root."""
    candidate = JOURNAL / name
    for path in (candidate, candidate.with_suffix(".tex")):
        if path.is_file():
            return path
    return None


def resolve_graphic(name: str) -> Path | None:
    candidate = JOURNAL / name
    if candidate.is_file():
        return candidate
    for suffix in GRAPHICS_SUFFIXES:
        if candidate.with_suffix(suffix).is_file():
            return candidate.with_suffix(suffix)
    return None


def flat_name(path: Path, taken: dict) -> str:
    """A unique flat basename for a source file, stable across runs."""
    relative = path.relative_to(JOURNAL)
    if relative in taken:
        return taken[relative]
    name = path.name
    if name in taken.values():
        prefix = "_".join(relative.parts[:-1])
        name = f"{prefix}_{path.name}" if prefix else path.name
    taken[relative] = name
    return name


def collect(root: Path, taken: dict, seen: set, files: dict) -> str:
    """Rewrite one .tex file, recursing into its \\input targets."""
    text = root.read_text(encoding="utf-8")

    def replace_input(match: re.Match) -> str:
        target = resolve_input(match.group(1))
        if target is None:
            print(f"  ! unresolved \\input{{{match.group(1)}}} in {root.name}", file=sys.stderr)
            return match.group(0)
        name = flat_name(target, taken)
        if target not in seen:
            seen.add(target)
            files[name] = collect(target, taken, seen, files)
        return "\\input{%s}" % Path(name).stem

    def replace_graphic(match: re.Match) -> str:
        target = resolve_graphic(match.group(2))
        if target is None:
            print(f"  ! unresolved graphic {{{match.group(2)}}} in {root.name}", file=sys.stderr)
            return match.group(0)
        name = flat_name(target, taken)
        files.setdefault(name, target)  # binary: carried as a Path
        return f"{match.group(1)}{name}{match.group(3)}"

    text = INPUT_RE.sub(replace_input, text)
    text = GRAPHICS_RE.sub(replace_graphic, text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--build", action="store_true",
                        help="test-compile the flattened directory with latexmk")
    args = parser.parse_args()

    main_tex = JOURNAL / "main.tex"
    if not main_tex.is_file():
        raise SystemExit(f"main.tex not found under {JOURNAL}")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    taken: dict = {}
    seen: set = {main_tex}
    files: dict = {}
    root_text = collect(main_tex, taken, seen, files)
    files["main.tex"] = root_text

    for name, payload in files.items():
        destination = out / name
        if isinstance(payload, Path):
            shutil.copy2(payload, destination)
        else:
            destination.write_text(payload, encoding="utf-8")

    # Bibliography: ship the compiled .bbl so no BibTeX pass is needed, and the
    # .bib alongside it for the production editor.
    for extra in ("main.bbl", "refs_journal.bib"):
        source = JOURNAL / extra
        if source.is_file():
            shutil.copy2(source, out / source.name)
        else:
            print(f"  ! {extra} missing -- build the manuscript first", file=sys.stderr)

    # Class and bibliography style, if locatable in the TeX installation.
    for asset in ("elsarticle.cls", "elsarticle-num.bst"):
        try:
            found = subprocess.run(["kpsewhich", asset], capture_output=True,
                                   text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            found = ""
        if found and Path(found).is_file():
            shutil.copy2(found, out / asset)
        else:
            print(f"  . {asset} not copied (available in the TeX distribution)")

    written = sorted(p.name for p in out.iterdir())
    print(f"\nflattened {len(written)} files into {out}")
    for name in written:
        print(f"  {name}")

    if args.build:
        print("\ntest-compiling ...")
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=out, capture_output=True, text=True)
        if result.returncode == 0 and (out / "main.pdf").is_file():
            print("build OK:", out / "main.pdf")
        else:
            print("build FAILED; see", out / "main.log", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
