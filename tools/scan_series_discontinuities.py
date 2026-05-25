#!/usr/bin/env python3
from __future__ import annotations

"""Scan configured symbols for large one-day moves in resolved price series."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qm_native_engine import DataStore, load_loose_json  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config")
    p.add_argument("--threshold", type=float, default=3.0, help="absolute one-day return threshold, e.g. 3.0 = 300%")
    p.add_argument("--symbols", default="", help="comma-separated symbols; default scans LetfMap tickers")
    args = p.parse_args()
    cfg = Path(args.config).expanduser().resolve()
    store = DataStore(cfg, use_testfolio_api=False, use_yahoo_fallback=False)
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = sorted(store.letf_map.keys())
    rows = []
    for sym in symbols:
        ps = store.load_symbol(sym)
        if ps is None:
            rows.append((sym, "MISSING", "", ""))
            continue
        worst_r = 0.0
        worst_d = ""
        for i in range(1, len(ps.values)):
            p0, p1 = ps.values[i-1], ps.values[i]
            if p0 is None or p1 is None or p0 == 0:
                continue
            r = p1 / p0 - 1.0
            if abs(r) > abs(worst_r):
                worst_r = r
                worst_d = store.market_days[i].isoformat()
        if abs(worst_r) >= args.threshold:
            rows.append((sym, worst_d, f"{worst_r:+.2%}", store.series_sources.get(sym, "")))
    print("Symbol,Date,Move,Source")
    for row in rows:
        print(",".join(str(x) for x in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
