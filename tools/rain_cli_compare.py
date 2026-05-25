#!/usr/bin/env python3
from __future__ import annotations

"""Run a local Rain CLI parity/export check.

Example:
    python tools/rain_cli_compare.py --input strategy.json --output rain_out --mode SYNTHETIC

This is optional. It requires a Java runtime new enough for the uploaded Rain
jar. The Python Streamlit app does not require Java for normal operation.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.rain_cli_tools import run_rain_cli  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Quantmage/Rain JSON strategy file")
    p.add_argument("--output", default="rain_cli_output", help="Rain CLI output directory")
    p.add_argument("--mode", choices=["NORMAL", "SYNTHETIC", "DEFAULT_PATHING"], default="SYNTHETIC")
    p.add_argument("--benchmark", default="")
    p.add_argument("--jar", default="", help="Optional explicit BacktestCLI.jar path")
    args = p.parse_args()
    res = run_rain_cli(
        Path(args.input).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        cli_jar=Path(args.jar).expanduser().resolve() if args.jar else None,
        mode=args.mode,
        benchmark=args.benchmark,
    )
    print("Command:", " ".join(res.command))
    print("Exit:", res.exit_code)
    if res.stdout:
        print("\nSTDOUT\n", res.stdout)
    if res.stderr:
        print("\nSTDERR\n", res.stderr)
    return 0 if res.ok else res.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
