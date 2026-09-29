#!/usr/bin/env python3
"""Run the HybridDB suite across interpreter/SQLite versions.

SQLite is not a pip dependency — it ships inside the interpreter — so the
version under test is whatever the chosen Python was built against. Users on a
newer Python therefore run a combination we never tested, which is exactly how
0.8.1/0.8.2 shipped a break against the chromadb floor (issue #1).

This gate runs the suite in isolated environments so each interpreter is tested
against its own bundled SQLite, and prints the resolved versions either way so a
failure can be attributed.

    python scripts/check_interpreter_matrix.py                 # default matrix
    python scripts/check_interpreter_matrix.py --python 3.13 --python 3.14
    python scripts/check_interpreter_matrix.py -- -k TestSQLiteCapabilities
    python scripts/check_interpreter_matrix.py --quick          # skip perf gates

Exits non-zero if any interpreter fails. The chromadb *floor* gate lives in
docs/RELEASE.md and is separate: this one covers the newest end of the range.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

DEFAULT_MATRIX = ("3.13", "3.14")

# Extras the suite needs. pytest-benchmark is required because the default
# `testpaths` collects tests/benchmarks.
EXTRAS = (
    "pytest",
    "pytest-asyncio",
    "pytest-benchmark",
    "pytest-timeout",
    "duckdb",
    "networkx",
    "scipy",
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _versions(python: str) -> str:
    """Print the interpreter's SQLite and chromadb versions."""
    code = (
        "import sqlite3;"
        "from chromadb.api.shared_system_client import SharedSystemClient as S;"
        "import chromadb, sys;"
        "print(f'    python {sys.version.split()[0]}  "
        "sqlite {sqlite3.sqlite_version}  chromadb {chromadb.__version__}  "
        "fts5_refcount={hasattr(S, \"_identifier_to_refcount\")}')"
    )
    result = subprocess.run(
        ["uv", "run", "--no-project", "--isolated", "--no-cache",
         "--python", python, *sum((["--with", e] for e in EXTRAS), []),
         "--with", ".", "python", "-c", code],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return (result.stdout or result.stderr).strip().splitlines()[-1] if (result.stdout or result.stderr) else "unknown"


def run_one(python: str, pytest_args: list[str]) -> tuple[str, bool, str]:
    print(f"-> python {python}")
    print(_versions(python))
    cmd = [
        "uv", "run", "--no-project", "--isolated", "--no-cache",
        "--python", python,
        *sum((["--with", e] for e in EXTRAS), []),
        "--with", ".", "python", "-m", "pytest", "-q", *pytest_args,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
    tail = ""
    if proc.returncode != 0:
        tail = " (see output above)"
    return python, proc.returncode == 0, tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="pythons", action="append",
                        help="interpreter to test (repeatable); default 3.13 + 3.14")
    parser.add_argument("--quick", action="store_true",
                        help="skip the wall-clock perf gates (faster, less signal)")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER,
                        help="arguments after -- are passed to pytest")
    args = parser.parse_args()

    pythons = args.pythons or list(DEFAULT_MATRIX)
    pytest_args = list(args.pytest_args)
    if args.pytest_args and args.pytest_args[0] == "--":
        pytest_args = args.pytest_args[1:]
    if args.quick:
        pytest_args += ["-k", "not rollback_beats_ingest and not update_heavy"]

    print("HybridDB interpreter/SQLite matrix")
    print("=" * 52)
    results = [run_one(p, pytest_args) for p in pythons]

    print()
    print("=" * 52)
    for python, ok, tail in results:
        print(f"{'PASS' if ok else 'FAIL'}  python {python}{tail}")
    failed = [p for p, ok, _ in results if not ok]
    if failed:
        print(f"\nFAILED on: {', '.join(failed)}")
        return 1
    print(f"\nAll {len(results)} interpreters passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
