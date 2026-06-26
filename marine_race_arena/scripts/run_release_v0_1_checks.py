"""Print or run the Marine Race Arena v0.1 release validation commands."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "-m", "compileall", "-q", "marine_race_arena", "tests"),
    ("conda", "run", "-n", "ocean", "python", "-m", "pytest", "-q"),
    ("conda", "run", "-n", "ocean", "python", "-m", "marine_race_arena.scripts.run_staggered_multi_rover_smoke"),
)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    commands = COMMANDS[:2] if args.skip_smoke else COMMANDS
    if args.print_only:
        _print_commands(commands)
        return 0
    return _run_commands(commands, repo_root=Path(__file__).resolve().parents[2])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        dest="print_only",
        action="store_false",
        help="Run the commands sequentially. The default only prints them.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the long HoloOcean staggered fleet smoke command.",
    )
    parser.set_defaults(print_only=True)
    return parser


def _print_commands(commands: tuple[tuple[str, ...], ...]) -> None:
    for command in commands:
        print(_format_command(command))


def _run_commands(commands: tuple[tuple[str, ...], ...], *, repo_root: Path) -> int:
    for command in commands:
        executable = command[0]
        if shutil.which(executable) is None:
            print(f"Missing executable: {executable}", file=sys.stderr)
            return 127
        print(f"Running: {_format_command(command)}")
        completed = subprocess.run(command, cwd=repo_root, check=False)
        if completed.returncode != 0:
            print(
                f"Command failed with exit code {completed.returncode}: {_format_command(command)}",
                file=sys.stderr,
            )
            return int(completed.returncode)
    return 0


def _format_command(command: tuple[str, ...]) -> str:
    return " ".join(_quote(part) for part in command)


def _quote(value: str) -> str:
    if any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
