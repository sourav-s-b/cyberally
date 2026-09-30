"""Dependency-free syntax check; this does not replace simulator or RL tests."""

import ast
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=root, check=True, capture_output=True,
    )
    paths = [p.decode("utf-8") for p in result.stdout.split(b"\0") if p]
    failed = []
    for relative in paths:
        path = root / relative
        try:
            ast.parse(path.read_bytes(), filename=relative)
        except (SyntaxError, UnicodeError) as error:
            failed.append(f"{relative}: {error}")
    if failed:
        print("\n".join(failed), file=sys.stderr)
        return 1
    if not paths:
        print("No tracked Python files; stage the source before checking.", file=sys.stderr)
        return 1
    print(f"Syntax OK: {len(paths)} tracked Python files. Runtime tests not performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
