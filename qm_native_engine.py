#!/usr/bin/env python3
"""
Quantmage-native local backtester using Testfolio cache files in ./config.

This does NOT call Composer APIs and does NOT require Rain's parser.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import io
import json
import math
import os
import re
import ssl
import statistics
import shutil
import struct
import tempfile
import threading
import time
import warnings
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import zipfile


# ----------------------------- Utility types -----------------------------


Date = dt.date


def parse_date_iso(value: str) -> Date:
    return dt.date.fromisoformat(value)


def format_date_mmddyyyy(d: Date) -> str:
    return d.strftime("%m/%d/%Y")


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def strip_json_comments(text: str) -> str:
    # Remove // comments in LetfMap-like files.
    return re.sub(r"(?m)//.*$", "", text)


def quote_object_keys(text: str) -> str:
    # Converts {ticker: "SPY"} -> {"ticker": "SPY"}
    return re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)


def load_loose_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    cooked = quote_object_keys(strip_json_comments(raw))
    # Allow trailing commas in pseudo-json files.
    cooked = re.sub(r",(\s*[}\]])", r"\1", cooked)
    return json.loads(cooked)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------- Price data model -----------------------------


@dataclass
class PriceSeries:
    symbol: str
    values: List[Optional[float]]

    def first_index(self) -> Optional[int]:
        for i, v in enumerate(self.values):
            if v is not None:
                return i
        return None

    def last_index(self) -> Optional[int]:
        for i in range(len(self.values) - 1, -1, -1):
            if self.values[i] is not None:
                return i
        return None


@dataclass
class DataStore:
    config_dir: Path
    use_testfolio_api: bool = True
    use_yahoo_fallback: bool = True
    allow_non_letf_proxy: bool = False
    testfolio_sim_jar: Optional[Path] = None
    refresh_sim_cache: bool = False
    extended_price_csvs: Tuple[Path, ...] = ()
    testfolio_cookie: str = ""
    testfolio_token: str = ""
    testfolio_api_url: str = "https://testfol.io/api/tactical"
    testfolio_timeout_sec: float = 40.0
    testfolio_retries: int = 2
    macro_map: Dict[str, str] = field(default_factory=lambda: {
        "@BAMLH0A0HYM2": "HYG",
        "@T10Y3M": "TLT",
        "@T10Y2Y": "TLT",
        "@VIXCLS": "UVXY",
        "@VXVCLS": "VIXM",
    })
    symbol_alias: Dict[str, str] = field(default_factory=lambda: {
        "^GSPC": "SPY",
        "^NDX": "QQQ",
        "^DXY": "UUP",
        "^BCOMGC": "GLD",
        "^VIX": "UVXY",
        "VIX": "UVXY",
    })
    testfolio_sim_alias: Dict[str, str] = field(default_factory=lambda: {
        "TBILL": "CASHX",
        "SPY": "SPYSIM",
        "OEF": "OEFSIM",
        "MDY": "MDYSIM",
        "IJR": "IJRSIM",
        "IWM": "IWMSIM",
        "KMLM": "KMLMSIM",
        "GLD": "GLDSIM",
        "SLV": "SLVSIM",
        "SVIX": "SVIXSIM",
        "TLT": "TLTSIM",
        "ZROZ": "ZROZSIM",
        "VXUS": "VXUSSIM",
        "VEA": "VEASIM",
        "VWO": "VWOSIM",
        "VSS": "VSSSIM",
        "EFV": "EFVSIM",
        "VTI": "VTISIM",
        "VT": "VTSIM",
        "DBMF": "DBMFSIM",
        "VIX": "VIXSIM",
        "GSG": "GSGSIM",
        "IEF": "IEFSIM",
        "IEI": "IEISIM",
        "SHY": "SHYSIM",
        "TIP": "TIPSIM",
        "BTC": "BTCSIM",
        "ETH": "ETHSIM",
        "XLB": "XLBSIM",
        "XLC": "XLCSIM",
        "XLE": "XLESIM",
        "XLF": "XLFSIM",
        "XLI": "XLISIM",
        "XLK": "XLKSIM",
        "XLP": "XLPSIM",
        "XLU": "XLUSIM",
        "XLV": "XLVSIM",
        "XLY": "XLYSIM",
        "QQQ": "QQQSIM",
        "CAOS": "CAOSSIM",
        "FNGU": "FNGUSIM",
        "MCI": "MCISIM",
        "GDE": "GDESIM",
        "RSSB": "RSSBSIM",
        "UUP": "UUPSIM",
        "VV": "VOOSIM",
        "VOO": "VOOSIM",
        "VTV": "VTVSIM",
        "VUG": "VUGSIM",
        "VO": "VOSIM",
        "VOE": "VOESIM",
        "VOT": "VOTSIM",
        "VB": "VBSIM",
        "VBR": "VBRSIM",
        "VBK": "VBKSIM",
        "IWC": "IWCSIM",
        "BND": "BNDSIM",
        "VNQ": "REITSIM",
    })
    symbol_proxy_map: Dict[str, str] = field(default_factory=lambda: {
        # Broad geography / style proxies
        "VT": "VTI",
        "VXUS": "VTI",
        "VEA": "VTI",
        "VWO": "VTI",
        "EFA": "VTI",
        "EEM": "VTI",
        "FXI": "VTI",
        "EWJ": "VTI",
        "INDA": "VTI",
        "IEV": "VTI",
        "SCHF": "VTI",
        "DGRO": "VTI",
        "USMV": "SPY",
        "SPLV": "SPY",
        "SPHQ": "SPY",
        "QUAL": "SPY",
        "RSP": "SPY",
        "SIZE": "SPY",
        "MTUM": "QQQ",
        "SCHD": "VTV",
        "RPV": "VTV",
        "SMH": "SOXX",
        "USDU": "UUP",
        # Bond / credit proxies
        "SPAB": "BND",
        "LQD": "BND",
        "ANGL": "BND",
        "BNDX": "BND",
        "AGG": "BND",
        "VGIT": "IEF",
        "VGLT": "TLT",
        "BLV": "TLT",
        "ZROZ": "TLT",
        "PULS": "SHV",
        # Vol / commodity proxies
        "UVIX": "UVXY",
        "SVIX": "SVXY",
        "SVOL": "SVXY",
        "VXZ": "VIXM",
        "USO": "XLE",
        "PDBC": "DBC",
        "NUGT": "GLD",
        "DUST": "GLL",
        # Single-name fallbacks
        "TSLA": "QQQ",
        "NVDA": "QQQ",
        "MSTR": "QQQ",
        "NVO": "VTV",
        "LLY": "VTV",
        "COST": "VTI",
        "PGR": "VTI",
        "WMT": "VTI",
        "WFC": "XLF",
        "XOM": "XLE",
    })
    stitch_map: Dict[str, str] = field(default_factory=lambda: {
        "BSV": "BND",
        "VOX": "XLC",
        "SVXY": "UVXY?L=-0.5",
        "VIXY": "UVXY?L=0.5",
        "VIXM": "UVXY?L=0.25",
    })

    market_days: List[Date] = field(init=False)
    day_to_index: Dict[Date, int] = field(init=False)
    epoch_to_index: Dict[int, int] = field(init=False)
    letf_map: Dict[str, Dict[str, Any]] = field(init=False)
    series_cache: Dict[str, PriceSeries] = field(default_factory=dict)
    proxied_symbols: Dict[str, str] = field(default_factory=dict)
    _loading_symbols: set[str] = field(default_factory=set)
    _testfolio_call_count: int = 0
    _testfolio_fail_count: int = 0
    _last_testfolio_call_at: float = 0.0
    _testfolio_lock: threading.Lock = field(default_factory=threading.Lock)
    _best_long_by_root: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _best_short_by_root: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _ssl_context: Optional[ssl.SSLContext] = None
    _testfolio_sim_path: Optional[Path] = None
    _testfolio_sim_cache_path: Optional[Path] = None
    _testfolio_sim_table: Optional[Dict[str, PriceSeries]] = None
    _external_csv_table: Optional[Dict[str, PriceSeries]] = None
    _testfolio_series_cache: Dict[str, Optional[PriceSeries]] = field(default_factory=dict)
    _yahoo_series_cache: Dict[str, Optional[PriceSeries]] = field(default_factory=dict)
    _yahoo_call_count: int = 0
    _yahoo_fail_count: int = 0
    _testfolio_auth_required: bool = False
    _testfolio_disabled_reason: str = ""

    def __post_init__(self) -> None:
        self.config_dir = self.config_dir.resolve()
        self.extended_price_csvs = self._resolve_extended_price_csvs(self.extended_price_csvs)
        self.market_days = self._load_market_days(self.config_dir / "MarketDays.dat")
        self.day_to_index = {d: i for i, d in enumerate(self.market_days)}
        self.epoch_to_index = {int((d - dt.date(1970, 1, 1)).days): i for i, d in enumerate(self.market_days)}
        self.letf_map = self._load_letf_map(self.config_dir / "LetfMap.json")
        self._best_long_by_root, self._best_short_by_root = self._build_best_letf_maps()
        self._ssl_context = self._build_ssl_context()
        self._testfolio_sim_path = self._resolve_testfolio_sim_jar(self.testfolio_sim_jar)
        self._testfolio_sim_cache_path = (self.config_dir / "extended_prices" / "testfolio-sim.csv").resolve()
        if self.refresh_sim_cache:
            self._refresh_testfolio_sim_cache()

    def data_source_summary(self) -> Dict[str, Any]:
        def count_dat(subdir: str) -> int:
            p = self.config_dir / subdir
            if not p.exists():
                return 0
            return sum(1 for _ in p.glob("*.dat"))

        mode = "testfolio_api+local_cache" if self.use_testfolio_api else "local_testfolio_cache"
        return {
            "type": mode,
            "config_dir": str(self.config_dir),
            "market_days_file": str(self.config_dir / "MarketDays.dat"),
            "letf_map_file": str(self.config_dir / "LetfMap.json"),
            "tickers_files": count_dat("tickers"),
            "ext_letf_files": count_dat("ext_letf"),
            "prices_files": count_dat("prices"),
            "extended_price_csvs": [str(p) for p in self.extended_price_csvs],
            "network_calls": self._testfolio_call_count > 0,
            "testfolio_sim_cache": str(self._testfolio_sim_cache_path) if self._testfolio_sim_cache_path else "",
            "testfolio_sim_cache_exists": bool(self._testfolio_sim_cache_path and self._testfolio_sim_cache_path.exists()),
            "testfolio_sim_jar": str(self._testfolio_sim_path) if self._testfolio_sim_path else "",
            "testfolio_sim_series": len(self._testfolio_sim_table or {}),
            "testfolio_api_enabled": self.use_testfolio_api,
            "testfolio_api_url": self.testfolio_api_url,
            "testfolio_auth_supplied": bool((self.testfolio_cookie or "").strip() or (self.testfolio_token or "").strip()),
            "testfolio_api_calls": self._testfolio_call_count,
            "testfolio_api_failures": self._testfolio_fail_count,
            "testfolio_auth_required": self._testfolio_auth_required,
            "testfolio_disabled_reason": self._testfolio_disabled_reason,
            "yahoo_fallback_enabled": self.use_yahoo_fallback,
            "yahoo_api_calls": self._yahoo_call_count,
            "yahoo_api_failures": self._yahoo_fail_count,
        }

    @staticmethod
    def _load_market_days(path: Path) -> List[Date]:
        b = path.read_bytes()
        out: List[Date] = []
        for i in range(0, len(b), 8):
            epoch_day = struct.unpack(">q", b[i : i + 8])[0]
            out.append(dt.date(1970, 1, 1) + dt.timedelta(days=epoch_day))
        return out

    @staticmethod
    def _load_letf_map(path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        rows = load_loose_json(path)
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                t = str(row.get("ticker", "")).upper().strip()
                if not t:
                    continue
                out[t] = row
        return out

    def _resolve_extended_price_csvs(self, paths: Tuple[Path, ...]) -> Tuple[Path, ...]:
        resolved: List[Path] = []
        for raw in paths:
            p = Path(raw).expanduser().resolve()
            if p.exists():
                resolved.append(p)
        default_dir = (self.config_dir / "extended_prices").resolve()
        if default_dir.exists():
            resolved.extend(sorted(default_dir.glob("*.csv")))
        deduped: List[Path] = []
        seen: set[Path] = set()
        for path in resolved:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return tuple(deduped)

    def _resolve_letf_root_and_effective_leverage(self, ticker: str) -> Tuple[Optional[str], Optional[float]]:
        row = self.letf_map.get(ticker)
        if not row:
            return None, None
        underlying = str(row.get("underlying", "")).upper().strip()
        if not underlying:
            return None, None
        eff = safe_float(row.get("leverage", 1.0), 1.0)
        root = underlying
        visited = {ticker}
        while root in self.letf_map and root not in visited:
            visited.add(root)
            nxt = self.letf_map[root]
            nxt_underlying = str(nxt.get("underlying", "")).upper().strip()
            if not nxt_underlying:
                break
            eff *= safe_float(nxt.get("leverage", 1.0), 1.0)
            root = nxt_underlying
        return root, eff

    def _build_best_letf_maps(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        best_long: Dict[str, Dict[str, Any]] = {}
        best_short: Dict[str, Dict[str, Any]] = {}
        for ticker in self.letf_map:
            root, eff = self._resolve_letf_root_and_effective_leverage(ticker)
            if not root or eff is None or not math.isfinite(eff) or eff == 0:
                continue
            meta = {
                "ticker": ticker,
                "root_underlying": root,
                "effective_leverage": eff,
            }
            if eff > 0:
                cur = best_long.get(root)
                if cur is None or eff > float(cur.get("effective_leverage", 0.0)):
                    best_long[root] = meta
            else:
                cur = best_short.get(root)
                if cur is None or abs(eff) > abs(float(cur.get("effective_leverage", 0.0))):
                    best_short[root] = meta
        return best_long, best_short

    def _safe_cutoff_index(self, row: Dict[str, Any]) -> Optional[int]:
        raw = row.get("cutoffDate")
        if raw is None:
            raw = row.get("cutoff_date")
        if not raw:
            return None
        try:
            d = parse_date_iso(str(raw))
        except Exception:
            return None
        return self.day_to_index.get(d)

    @staticmethod
    def _build_ssl_context() -> Optional[ssl.SSLContext]:
        try:
            import certifi  # type: ignore

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            try:
                return ssl.create_default_context()
            except Exception:
                return None

    @staticmethod
    def _resolve_testfolio_sim_jar(path: Optional[Path]) -> Optional[Path]:
        candidates: List[Path] = []
        if path:
            candidates.append(Path(path).expanduser())
        candidates.extend([
            Path.cwd() / "BacktestReport.jar",
            Path.home() / "Downloads" / "BacktestReport.jar",
        ])

        existing: List[Path] = []
        seen: set[str] = set()
        for cand in candidates:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.exists() and cand.is_file():
                existing.append(cand.resolve())
        if not existing:
            return None
        return max(existing, key=lambda p: p.stat().st_mtime)

    def _read_wide_price_csv(self, path: Path) -> Dict[str, PriceSeries]:
        out: Dict[str, PriceSeries] = {}
        if not path.exists() or not path.is_file():
            return out
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header or len(header) < 2:
                    return out
                symbols = [str(col or "").strip().upper() for col in header[1:]]
                columns = {sym: [None] * len(self.market_days) for sym in symbols if sym}
                for row in reader:
                    if not row:
                        continue
                    raw_date = str(row[0] or "").strip()
                    if not raw_date:
                        continue
                    try:
                        idx = self.day_to_index[parse_date_iso(raw_date)]
                    except Exception:
                        continue
                    for col_idx, sym in enumerate(symbols, start=1):
                        if not sym or col_idx >= len(row):
                            continue
                        raw_val = str(row[col_idx] or "").strip()
                        if not raw_val:
                            continue
                        try:
                            px = float(raw_val)
                        except Exception:
                            continue
                        if math.isfinite(px) and px > 0:
                            columns[sym][idx] = px
                out = {sym: PriceSeries(symbol=sym, values=vals) for sym, vals in columns.items()}
        except Exception:
            return {}
        return out

    def _refresh_testfolio_sim_cache(self) -> bool:
        jar_path = self._testfolio_sim_path
        cache_path = self._testfolio_sim_cache_path or (self.config_dir / "extended_prices" / "testfolio-sim.csv")
        if jar_path is None or not jar_path.exists():
            return False
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(jar_path) as zf:
                with zf.open("Testfolio_SIM.csv") as src, cache_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            return True
        except Exception:
            return False

    def _load_testfolio_sim_table(self) -> Dict[str, PriceSeries]:
        if self._testfolio_sim_table is not None:
            return self._testfolio_sim_table

        out: Dict[str, PriceSeries] = {}
        cache_path = self._testfolio_sim_cache_path or (self.config_dir / "extended_prices" / "testfolio-sim.csv")
        # Prefer persistent cache; it removes fragile runtime jar dependency.
        if cache_path.exists():
            out = self._read_wide_price_csv(cache_path)
            if out:
                self._testfolio_sim_table = out
                return out

        jar_path = self._testfolio_sim_path
        if jar_path is not None:
            try:
                with zipfile.ZipFile(jar_path) as zf:
                    with zf.open("Testfolio_SIM.csv") as raw:
                        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                        reader = csv.reader(text)
                        header = next(reader, None)
                        if header and len(header) >= 2:
                            symbols = [str(col or "").strip().upper() for col in header[1:]]
                            columns = {sym: [None] * len(self.market_days) for sym in symbols if sym}
                            for row in reader:
                                if not row:
                                    continue
                                raw_date = str(row[0] or "").strip()
                                if not raw_date:
                                    continue
                                try:
                                    idx = self.day_to_index[parse_date_iso(raw_date)]
                                except Exception:
                                    continue
                                for col_idx, sym in enumerate(symbols, start=1):
                                    if not sym or col_idx >= len(row):
                                        continue
                                    raw_val = str(row[col_idx] or "").strip()
                                    if not raw_val:
                                        continue
                                    try:
                                        px = float(raw_val)
                                    except Exception:
                                        continue
                                    if math.isfinite(px) and px > 0:
                                        columns[sym][idx] = px
                            out = {sym: PriceSeries(symbol=sym, values=vals) for sym, vals in columns.items()}
                            if out and not cache_path.exists():
                                self._refresh_testfolio_sim_cache()
            except Exception:
                out = {}

        self._testfolio_sim_table = out
        return out

    def _testfolio_sim_candidates(self, symbol: str) -> List[str]:
        candidates = [symbol]
        alias = self.testfolio_sim_alias.get(symbol)
        if alias:
            candidates.append(alias)
        if symbol != "CASHX":
            candidates.append(f"{symbol}SIM")
        return list(dict.fromkeys(c for c in candidates if c))

    def _fetch_testfolio_sim_history(self, symbol: str) -> Optional[PriceSeries]:
        table = self._load_testfolio_sim_table()
        if not table:
            return None
        for candidate in self._testfolio_sim_candidates(symbol):
            series = table.get(candidate)
            if series is None:
                continue
            return PriceSeries(symbol=symbol, values=series.values)
        return None

    def _load_external_csv_table(self) -> Dict[str, PriceSeries]:
        if self._external_csv_table is not None:
            return self._external_csv_table

        columns: Dict[str, List[Optional[float]]] = {}
        for path in self.extended_price_csvs:
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames or "Date" not in reader.fieldnames:
                        continue
                    symbols = [str(name or "").strip().upper() for name in reader.fieldnames if name and name != "Date"]
                    for sym in symbols:
                        if sym and sym not in columns:
                            columns[sym] = [None] * len(self.market_days)
                    for row in reader:
                        raw_date = str(row.get("Date", "")).strip()
                        if not raw_date:
                            continue
                        try:
                            date = parse_date_iso(raw_date)
                        except Exception:
                            continue
                        idx = self.day_to_index.get(date)
                        if idx is None:
                            continue
                        for sym in symbols:
                            raw_val = row.get(sym)
                            if raw_val in (None, ""):
                                continue
                            try:
                                px = float(raw_val)
                            except Exception:
                                continue
                            if not math.isfinite(px) or px <= 0:
                                continue
                            columns[sym][idx] = px
            except Exception:
                continue

        self._external_csv_table = {sym: PriceSeries(symbol=sym, values=vals) for sym, vals in columns.items()}
        return self._external_csv_table

    def _fetch_external_csv_history(self, symbol: str) -> Optional[PriceSeries]:
        table = self._load_external_csv_table()
        if not table:
            return None
        for candidate in self._testfolio_sim_candidates(symbol):
            series = table.get(candidate)
            if series is not None:
                return PriceSeries(symbol=symbol, values=series.values)
        return None

    def _merge_fill_missing(
        self,
        symbol: str,
        base: Optional[PriceSeries],
        extra: Optional[PriceSeries],
        cutoff_idx: Optional[int] = None,
    ) -> Optional[PriceSeries]:
        if extra is None:
            return base
        if base is None:
            return extra

        ratios: List[float] = []
        for i, ev in enumerate(extra.values):
            if cutoff_idx is not None and i >= cutoff_idx:
                continue
            bv = base.values[i]
            if bv is None or ev is None or ev == 0:
                continue
            r = bv / ev
            if r > 0 and math.isfinite(r):
                ratios.append(r)
                if len(ratios) > 2000:
                    ratios.pop(0)
        scale = statistics.median(ratios) if ratios else 1.0

        vals = list(base.values)
        for i, ev in enumerate(extra.values):
            if cutoff_idx is not None and i >= cutoff_idx:
                continue
            if vals[i] is None and ev is not None:
                vals[i] = ev * scale
        return PriceSeries(symbol=symbol, values=vals)

    def _throttle_testfolio(self) -> None:
        with self._testfolio_lock:
            now = time.time()
            elapsed = now - self._last_testfolio_call_at
            if elapsed < 2.0:
                time.sleep(max(0.1, 2.0 - elapsed))
            self._last_testfolio_call_at = time.time()

    def _testfolio_payload(self, symbol: str, start_date: str = "", start_val: int = 10000) -> Dict[str, Any]:
        req_symbol = symbol
        return {
            "name": "STRATEGY",
            "start_date": start_date,
            "end_date": "",
            "start_val": int(start_val),
            "trading_cost": 0,
            "rolling_window": 60,
            "signals": [],
            "trading_freq": "Daily",
            "allocations": [
                {
                    "name": req_symbol,
                    "signals": [],
                    "ops": [],
                    "nots": [],
                    "tickers": [{"ticker": req_symbol, "percent": 100}],
                    "drag": 0,
                }
            ],
        }

    def _parse_testfolio_history(self, req_symbol: str, payload: Dict[str, Any]) -> Optional[PriceSeries]:
        detail = str(payload.get("detail") or "").strip()
        if detail:
            dlow = detail.lower()
            if "sign in" in dlow or "login" in dlow or "account" in dlow:
                self._testfolio_auth_required = True
                self._testfolio_disabled_reason = detail
            return None

        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            return None

        charts = payload.get("charts")
        if not isinstance(charts, dict):
            return None
        history = charts.get("history")
        stats = payload.get("stats")
        if not isinstance(history, list) or not history:
            return None
        if not isinstance(stats, list):
            return None

        target_names = {
            req_symbol.upper(),
            req_symbol.replace("/", "-").upper(),
            req_symbol.replace("-", "/").upper(),
        }
        hit_idx: Optional[int] = None
        for i, row in enumerate(stats):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").upper()
            if name in target_names:
                hit_idx = i
                break
        if hit_idx is None:
            return None

        row_idx = hit_idx + 1
        if row_idx >= len(history):
            return None
        epochs = history[0]
        prices = history[row_idx]
        if not isinstance(epochs, list) or not isinstance(prices, list):
            return None

        vals: List[Optional[float]] = [None] * len(self.market_days)
        any_hit = False
        for e, v in zip(epochs, prices):
            try:
                px = float(v)
            except Exception:
                continue
            if px < 100.0 or px > 1.0e12:
                break
            try:
                ts = int(e)
            except Exception:
                continue
            d = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
            idx = self.day_to_index.get(d)
            if idx is None:
                continue
            vals[idx] = px
            any_hit = True
        if not any_hit:
            return None
        return PriceSeries(symbol=req_symbol, values=vals)

    def _fetch_testfolio_history(self, req_symbol: str) -> Optional[PriceSeries]:
        if not self.use_testfolio_api:
            return None
        if self._testfolio_auth_required:
            return None
        key = req_symbol.strip().upper()
        if key in self._testfolio_series_cache:
            return self._testfolio_series_cache[key]
        body = json.dumps(self._testfolio_payload(req_symbol), separators=(",", ":")).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(max(0, self.testfolio_retries) + 1):
            try:
                self._throttle_testfolio()
                req = urllib.request.Request(
                    self.testfolio_api_url,
                    data=body,
                    headers=self._testfolio_headers(),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.testfolio_timeout_sec, context=self._ssl_context) as resp:
                    raw = resp.read()
                self._testfolio_call_count += 1
                payload = json.loads(raw.decode("utf-8"))
                parsed = self._parse_testfolio_history(req_symbol, payload)
                self._testfolio_series_cache[key] = parsed
                return parsed
            except urllib.error.HTTPError as ex:
                self._testfolio_fail_count += 1
                last_err = ex
                if ex.code in (401, 403):
                    self._testfolio_auth_required = True
                    if (self.testfolio_cookie or "").strip() or (self.testfolio_token or "").strip():
                        self._testfolio_disabled_reason = f"HTTP {ex.code}: supplied Testfolio auth cookie/token is invalid or expired"
                    else:
                        self._testfolio_disabled_reason = f"HTTP {ex.code}: login required or endpoint blocked"
                    break
                if ex.code == 429 and attempt < self.testfolio_retries:
                    time.sleep(3.0)
                    continue
                break
            except Exception as ex:  # network/dns/json issues
                self._testfolio_fail_count += 1
                last_err = ex
                if attempt < self.testfolio_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
        if last_err is not None:
            self._testfolio_series_cache[key] = None
            return None
        self._testfolio_series_cache[key] = None
        return None

    def _testfolio_headers(self) -> Dict[str, str]:
        headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "Mozilla/5.0",
        }
        cookie = (self.testfolio_cookie or "").strip()
        if cookie:
            headers["Cookie"] = cookie
        token = (self.testfolio_token or "").strip()
        if token:
            if token.lower().startswith("bearer "):
                headers["Authorization"] = token
            else:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _parse_yahoo_chart(self, symbol: str, payload: Dict[str, Any]) -> Optional[PriceSeries]:
        chart = payload.get("chart")
        if not isinstance(chart, dict):
            return None
        result = chart.get("result")
        if not isinstance(result, list) or not result:
            return None
        r0 = result[0]
        if not isinstance(r0, dict):
            return None
        timestamps = r0.get("timestamp")
        indicators = r0.get("indicators")
        if not isinstance(timestamps, list) or not isinstance(indicators, dict):
            return None
        adj = None
        adjclose = indicators.get("adjclose")
        if isinstance(adjclose, list) and adjclose and isinstance(adjclose[0], dict):
            adj = adjclose[0].get("adjclose")
        close = None
        quote = indicators.get("quote")
        if isinstance(quote, list) and quote and isinstance(quote[0], dict):
            close = quote[0].get("close")
        prices = adj if isinstance(adj, list) else close
        if not isinstance(prices, list):
            return None

        vals: List[Optional[float]] = [None] * len(self.market_days)
        any_hit = False
        for ts, px in zip(timestamps, prices):
            if px is None:
                continue
            try:
                d = dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date()
                v = float(px)
            except Exception:
                continue
            if not math.isfinite(v) or v <= 0:
                continue
            idx = self.day_to_index.get(d)
            if idx is None:
                continue
            vals[idx] = v
            any_hit = True
        if not any_hit:
            return None
        return PriceSeries(symbol=symbol, values=vals)

    def _fetch_yahoo_history(self, symbol: str) -> Optional[PriceSeries]:
        if not self.use_yahoo_fallback:
            return None
        key = symbol.strip().upper()
        if key in self._yahoo_series_cache:
            return self._yahoo_series_cache[key]

        candidates = [symbol]
        if "." in symbol:
            candidates.append(symbol.replace(".", "-"))
        if "/" in symbol:
            candidates.append(symbol.replace("/", "-"))
        candidates = list(dict.fromkeys(candidates))

        for cand in candidates:
            enc = urllib.parse.quote(cand, safe="^=-")
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
                f"?period1=0&period2={int(time.time())}&interval=1d&events=history&includeAdjustedClose=true"
            )
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0",
                },
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.testfolio_timeout_sec, context=self._ssl_context) as resp:
                    raw = resp.read()
                self._yahoo_call_count += 1
                payload = json.loads(raw.decode("utf-8"))
                parsed = self._parse_yahoo_chart(symbol, payload)
                if parsed is not None:
                    self._yahoo_series_cache[key] = parsed
                    return parsed
            except Exception:
                self._yahoo_fail_count += 1
                continue

        self._yahoo_series_cache[key] = None
        return None

    def _letf_request_symbol(self, row: Dict[str, Any]) -> Optional[str]:
        underlying = str(row.get("underlying", "")).upper().strip()
        if not underlying:
            return None
        out = underlying
        lev = safe_float(row.get("leverage", 1.0), 1.0)
        if math.isfinite(lev) and lev != 1.0:
            lev_str = str(lev)
            if lev_str.endswith(".0"):
                lev_str = lev_str[:-2]
            out += f"?L={lev_str}"
        exp = safe_float(row.get("expenseRatio", 0.0), 0.0)
        if math.isfinite(exp) and exp != 0.0:
            out += ("&" if "?" in out else "?") + f"E={exp}"
        sw = safe_float(row.get("swapExposure", 0.0), 0.0)
        if math.isfinite(sw) and sw != 0.0:
            out += ("&" if "?" in out else "?") + f"SW={sw}"
        return out

    def _extend_with_testfolio_api(self, symbol: str, base: Optional[PriceSeries]) -> Optional[PriceSeries]:
        if not self.use_testfolio_api:
            return base
        row = self.letf_map.get(symbol) or {}
        cutoff_idx = self._safe_cutoff_index(row) if row else None

        merged = self._merge_fill_missing(symbol, base, self._fetch_testfolio_history(symbol), cutoff_idx)
        req_symbol = self._letf_request_symbol(row) if row else None
        if req_symbol and req_symbol != symbol:
            merged = self._merge_fill_missing(symbol, merged, self._fetch_testfolio_history(req_symbol), cutoff_idx)
        return merged

    def _read_dat_file(self, path: Path, symbol: str) -> PriceSeries:
        b = path.read_bytes()
        if len(b) < 16:
            return PriceSeries(symbol=symbol, values=[None] * len(self.market_days))

        start_epoch, end_epoch = struct.unpack(">qq", b[:16])
        if start_epoch not in self.epoch_to_index or end_epoch not in self.epoch_to_index:
            raise ValueError(f"{path}: date range not in MarketDays.dat")
        i0 = self.epoch_to_index[start_epoch]
        i1 = self.epoch_to_index[end_epoch]
        n = i1 - i0 + 1

        expected = 16 + n * 8
        if expected != len(b):
            raise ValueError(f"{path}: malformed length {len(b)} (expected {expected})")

        vals: List[Optional[float]] = [None] * len(self.market_days)
        off = 16
        for i in range(n):
            vals[i0 + i] = struct.unpack(">d", b[off : off + 8])[0]
            off += 8
        return PriceSeries(symbol=symbol, values=vals)

    def _merge_series(self, symbol: str, parts: List[PriceSeries]) -> PriceSeries:
        vals: List[Optional[float]] = [None] * len(self.market_days)
        for pi, part in enumerate(parts):
            scale = 1.0
            if pi > 0:
                # Different cache folders can encode the same price history with
                # different nominal scales (e.g. 10000-based vs market-price).
                # Normalize later sources to the existing merged scale to avoid
                # artificial return shocks at splice boundaries.
                ratios: List[float] = []
                for i, pv in enumerate(part.values):
                    mv = vals[i]
                    if mv is None or pv is None or pv == 0:
                        continue
                    r = mv / pv
                    if r > 0 and math.isfinite(r):
                        ratios.append(r)
                        if len(ratios) >= 2000:
                            break
                if ratios:
                    scale = statistics.median(ratios)
                else:
                    fi = part.first_index()
                    if fi is not None:
                        lo = max(0, fi - 5)
                        hi = min(len(vals), fi + 6)
                        for i in range(lo, hi):
                            mv = vals[i]
                            pv = part.values[i]
                            if mv is None or pv is None or pv == 0:
                                continue
                            r = mv / pv
                            if r > 0 and math.isfinite(r):
                                scale = r
                                break
                        if scale == 1.0 and fi > 0:
                            mv = vals[fi - 1]
                            pv = part.values[fi]
                            if mv is not None and pv is not None and pv != 0:
                                r = mv / pv
                                if r > 0 and math.isfinite(r):
                                    scale = r
            for i, v in enumerate(part.values):
                if v is not None:
                    vals[i] = v * scale
        return PriceSeries(symbol=symbol, values=vals)

    def _load_direct_symbol(self, symbol: str) -> Optional[PriceSeries]:
        # Priority: historical tickers (long) -> ext_letf -> prices (recent override)
        locations = [
            self.config_dir / "tickers" / f"{symbol}.dat",
            self.config_dir / "ext_letf" / f"{symbol}.dat",
            self.config_dir / "prices" / f"{symbol}.dat",
        ]
        parts: List[PriceSeries] = []
        for p in locations:
            if p.exists():
                parts.append(self._read_dat_file(p, symbol))
        if not parts:
            return None
        return self._merge_series(symbol, parts)

    def _build_ratio_series(self, symbol: str, lhs: str, rhs: str) -> Optional[PriceSeries]:
        l = self.load_symbol(lhs)
        r = self.load_symbol(rhs)
        if l is None or r is None:
            return None
        vals: List[Optional[float]] = [None] * len(self.market_days)
        for i in range(len(vals)):
            lv = l.values[i]
            rv = r.values[i]
            if lv is None or rv is None or rv == 0:
                continue
            vals[i] = lv / rv
        return PriceSeries(symbol=symbol, values=vals)

    def _build_levered_view(self, symbol: str, base_symbol: str, leverage: float) -> Optional[PriceSeries]:
        base = self.load_symbol(base_symbol)
        if base is None:
            return None

        vals: List[Optional[float]] = [None] * len(self.market_days)
        first = base.first_index()
        if first is None:
            return PriceSeries(symbol=symbol, values=vals)

        vals[first] = base.values[first]
        for i in range(first + 1, len(vals)):
            b0 = base.values[i - 1]
            b1 = base.values[i]
            prev = vals[i - 1]
            if b0 is None or b1 is None or prev is None or b0 == 0:
                vals[i] = None
                continue
            r = (b1 / b0) - 1.0
            vals[i] = prev * (1.0 + leverage * r)
        return PriceSeries(symbol=symbol, values=vals)

    def _extend_with_letf_map(self, symbol: str, direct: Optional[PriceSeries]) -> Optional[PriceSeries]:
        row = self.letf_map.get(symbol)
        if not row:
            return direct

        underlying = str(row.get("underlying", "")).upper().strip()
        lev = safe_float(row.get("leverage", 1.0), 1.0)
        if not underlying:
            return direct

        # Build synthetic from underlying and leverage.
        synthetic = self._build_levered_view(symbol, underlying, lev)
        if synthetic is None:
            return direct

        if direct is None:
            return synthetic
        # Scale-align synthetic LETF history to the direct series before filling
        # any gaps. This avoids large nominal jumps when direct history ends
        # or has holes inside the live range.
        return self._merge_fill_missing(symbol, direct, synthetic)

    def _extend_with_stitch_map(self, symbol: str, base: Optional[PriceSeries]) -> Optional[PriceSeries]:
        proxy = str(self.stitch_map.get(symbol, "")).upper().strip()
        if not proxy or proxy == symbol:
            return base
        proxy_series = self.load_symbol(proxy)
        if proxy_series is None:
            return base
        merged = self._merge_fill_missing(symbol, base, proxy_series)
        proxy_first = proxy_series.first_index()
        base_first = base.first_index() if base is not None else None
        if merged is not None and proxy_first is not None and (base_first is None or proxy_first < base_first):
            self.proxied_symbols[symbol] = proxy
        return merged

    def _heuristic_proxy(self, symbol: str) -> Optional[str]:
        if symbol in self.symbol_proxy_map:
            return self.symbol_proxy_map[symbol]

        # Generic class-level heuristics for unknown assets.
        if re.search(r"(BND|BOND|LQD|AGG|SHY|IEF|TLT|TREAS)", symbol):
            return "BND"
        if re.search(r"(VIX|VOL)", symbol):
            return "VIXM"
        if re.search(r"(GLD|SLV|GDX|GOLD|SILV)", symbol):
            return "GLD"
        if re.search(r"(OIL|USO|UNG|GAS|XOP|FCG|DBC|DBA|DBO|PDBC|CPER|XME)", symbol):
            return "DBC"
        if symbol in {"DIA", "IWM", "MDY", "RUT", "RUA", "VTWO", "IJR"}:
            return "SPY"
        if symbol in {"BRK.B", "BRKB"}:
            return "VTI"

        # Final fallback for unknown US ticker-like symbols.
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol):
            return "SPY"
        return None

    def _letf_replacement_proxy(self, symbol: str) -> Optional[str]:
        row = self.letf_map.get(symbol)
        if not row:
            return None
        root, eff = self._resolve_letf_root_and_effective_leverage(symbol)
        if not root or eff is None or not math.isfinite(eff) or eff == 0:
            underlying = str(row.get("underlying", "")).upper().strip()
            return underlying or None
        if eff > 0:
            best = self._best_long_by_root.get(root)
        else:
            best = self._best_short_by_root.get(root)
        if best:
            candidate = str(best.get("ticker") or "").upper().strip()
            if candidate and candidate != symbol:
                return candidate
        underlying = str(row.get("underlying", "")).upper().strip()
        return underlying or None

    def load_symbol(self, raw_symbol: str) -> Optional[PriceSeries]:
        symbol = raw_symbol.strip().upper()
        if not symbol:
            return None

        # Macro proxies
        if symbol in self.macro_map:
            symbol = self.macro_map[symbol]
        if symbol in self.symbol_alias:
            symbol = self.symbol_alias[symbol]

        if symbol in self.series_cache:
            return self.series_cache[symbol]

        if symbol in self._loading_symbols:
            return None
        self._loading_symbols.add(symbol)
        try:
            # Dynamic leverage syntax: XYZ?L=0.75
            qmatch = re.fullmatch(r"([^?]+)\?L=([-+]?[0-9]*\.?[0-9]+)", symbol)
            if qmatch:
                base = qmatch.group(1).upper().strip()
                lev = safe_float(qmatch.group(2), 1.0)
                s = self._build_levered_view(symbol, base, lev)
                if s is not None:
                    self.series_cache[symbol] = s
                return s

            # Ratio symbol like AAPL/AMZN
            if "/" in symbol and symbol not in self.letf_map:
                lhs, rhs = symbol.split("/", 1)
                s = self._build_ratio_series(symbol, lhs.strip().upper(), rhs.strip().upper())
                if s is not None:
                    self.series_cache[symbol] = s
                return s

            direct = self._load_direct_symbol(symbol)
            merged = self._extend_with_letf_map(symbol, direct)
            # Prefer local extended-price/SIM cache before jar or network fallback.
            merged = self._merge_fill_missing(symbol, merged, self._fetch_external_csv_history(symbol))
            merged = self._merge_fill_missing(symbol, merged, self._fetch_testfolio_sim_history(symbol))
            merged = self._extend_with_stitch_map(symbol, merged)
            merged = self._extend_with_testfolio_api(symbol, merged)
            merged = self._merge_fill_missing(symbol, merged, self._fetch_yahoo_history(symbol))
            if merged is not None:
                self.series_cache[symbol] = merged
                return merged

            letf_proxy = self._letf_replacement_proxy(symbol)
            if letf_proxy and letf_proxy != symbol:
                base = self.load_symbol(letf_proxy)
                if base is not None:
                    proxied = PriceSeries(symbol=symbol, values=base.values)
                    self.series_cache[symbol] = proxied
                    self.proxied_symbols[symbol] = letf_proxy
                    return proxied

            if self.allow_non_letf_proxy:
                proxy = self._heuristic_proxy(symbol)
                if proxy and proxy != symbol:
                    base = self.load_symbol(proxy)
                    if base is not None:
                        # Reuse the same underlying value array to avoid duplicated memory/work.
                        proxied = PriceSeries(symbol=symbol, values=base.values)
                        self.series_cache[symbol] = proxied
                        self.proxied_symbols[symbol] = proxy
                        return proxied

            return None
        finally:
            self._loading_symbols.discard(symbol)

    def daily_return(self, symbol: str, idx: int) -> Optional[float]:
        s = self.load_symbol(symbol)
        if s is None or idx <= 0 or idx >= len(self.market_days):
            return None
        p0 = s.values[idx - 1]
        p1 = s.values[idx]
        if p0 is None or p1 is None or p0 == 0:
            return None
        return (p1 / p0) - 1.0


# ----------------------------- Indicator engine -----------------------------


class IndicatorEngine:
    def __init__(self, store: DataStore):
        self.store = store
        self.cache: Dict[Tuple[str, str, int], List[Optional[float]]] = {}

    def series(self, symbol: str) -> Optional[PriceSeries]:
        return self.store.load_symbol(symbol)

    def indicator_array(self, ind_type: str, symbol: str, window: int) -> List[Optional[float]]:
        sym = symbol.strip().upper()
        sym = self.store.macro_map.get(sym, sym)
        sym = self.store.symbol_alias.get(sym, sym)
        # Populate any proxy mapping first, then cache by canonical symbol.
        self.store.load_symbol(sym)
        canonical = self.store.proxied_symbols.get(sym, sym)
        key = (ind_type, canonical, window)
        if key in self.cache:
            return self.cache[key]
        arr = self._compute_indicator(ind_type, canonical, window)
        self.cache[key] = arr
        return arr

    def indicator_value(self, ind_type: str, symbol: str, window: int, idx: int) -> Optional[float]:
        arr = self.indicator_array(ind_type, symbol, window)
        if idx < 0 or idx >= len(arr):
            return None
        return arr[idx]

    def _compute_indicator(self, ind_type: str, symbol: str, window: int) -> List[Optional[float]]:
        s = self.series(symbol)
        n = len(self.store.market_days)
        out: List[Optional[float]] = [None] * n
        if s is None:
            return out

        vals = s.values

        def price(i: int) -> Optional[float]:
            if i < 0 or i >= n:
                return None
            return vals[i]

        # Precompute returns where needed.
        rets: List[Optional[float]] = [None] * n
        for i in range(1, n):
            p0 = vals[i - 1]
            p1 = vals[i]
            if p0 is None or p1 is None or p0 == 0:
                continue
            rets[i] = (p1 / p0) - 1.0

        w = max(1, int(window))

        if ind_type == "CurrentPrice":
            return vals.copy()

        if ind_type == "CumulativeReturn":
            for i in range(n):
                p_now = price(i)
                p_old = price(i - w)
                if p_now is None or p_old is None or p_old == 0:
                    continue
                out[i] = ((p_now / p_old) - 1.0) * 100.0
            return out

        if ind_type == "RelativeStrengthIndex":
            # Wilder RSI (RMA smoothing), aligned with common TA implementations.
            period = w
            gains: List[float] = []
            losses: List[float] = []
            avg_gain: Optional[float] = None
            avg_loss: Optional[float] = None
            for i in range(1, n):
                r = rets[i]
                if r is None:
                    continue
                gain = max(r, 0.0)
                loss = max(-r, 0.0)
                if avg_gain is None or avg_loss is None:
                    gains.append(gain)
                    losses.append(loss)
                    if len(gains) < period:
                        continue
                    avg_gain = sum(gains[-period:]) / float(period)
                    avg_loss = sum(losses[-period:]) / float(period)
                else:
                    avg_gain = ((avg_gain * (period - 1.0)) + gain) / float(period)
                    avg_loss = ((avg_loss * (period - 1.0)) + loss) / float(period)
                if avg_loss == 0:
                    out[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    out[i] = 100.0 - (100.0 / (1.0 + rs))
            return out

        if ind_type == "Volatility":
            for i in range(n):
                ws = []
                lo = max(1, i - w + 1)
                for j in range(lo, i + 1):
                    r = rets[j]
                    if r is not None:
                        ws.append(r)
                if len(ws) >= 2:
                    out[i] = statistics.pstdev(ws) * 100.0
            return out

        if ind_type in {"13612wMomentum", "13612uMomentum", "Sma12Momentum"}:
            m1, m3, m6, m12 = 21, 63, 126, 252
            for i in range(n):
                p0 = price(i)
                p1 = price(i - m1)
                p3 = price(i - m3)
                p6 = price(i - m6)
                p12 = price(i - m12)
                if None in (p0, p1, p3, p6, p12):
                    continue
                assert p0 is not None and p1 is not None and p3 is not None and p6 is not None and p12 is not None
                if min(p1, p3, p6, p12) == 0:
                    continue
                if ind_type == "13612wMomentum":
                    out[i] = (
                        12.0 * ((p0 / p1) - 1.0)
                        + 4.0 * ((p0 / p3) - 1.0)
                        + 2.0 * ((p0 / p6) - 1.0)
                        + 1.0 * ((p0 / p12) - 1.0)
                    ) / 19.0 * 100.0
                elif ind_type == "13612uMomentum":
                    out[i] = (
                        ((p0 / p1) - 1.0)
                        + ((p0 / p3) - 1.0)
                        + ((p0 / p6) - 1.0)
                        + ((p0 / p12) - 1.0)
                    ) / 4.0 * 100.0
                else:
                    # SMA12 momentum: 13*p0/(p0+p1+...+p12)-1 using monthly samples.
                    months = [
                        price(i - 21 * k)
                        for k in range(13)
                    ]
                    if any(v is None or v == 0 for v in months):
                        continue
                    denom = sum(v for v in months if v is not None)
                    if denom == 0:
                        continue
                    out[i] = (13.0 * p0 / denom - 1.0) * 100.0
            return out

        if ind_type == "MovingAverageOfReturns":
            for i in range(n):
                lo = max(1, i - w + 1)
                ws = [rets[j] for j in range(lo, i + 1) if rets[j] is not None]
                if ws:
                    out[i] = (sum(ws) / len(ws)) * 100.0
            return out

        if ind_type == "MovingAverage":
            for i in range(n):
                lo = max(0, i - w + 1)
                ws = [vals[j] for j in range(lo, i + 1) if vals[j] is not None]
                if ws:
                    out[i] = sum(ws) / len(ws)
            return out

        if ind_type == "ExponentialMovingAverage":
            alpha = 2.0 / (w + 1.0)
            ema: Optional[float] = None
            for i in range(n):
                p = vals[i]
                if p is None:
                    continue
                if ema is None:
                    ema = p
                else:
                    ema = (1 - alpha) * ema + alpha * p
                out[i] = ema
            return out

        if ind_type == "UltimateSmoother":
            # Ehlers-style 2-pole smoother approximation.
            period = max(2, w)
            a1 = math.exp(-math.sqrt(2.0) * math.pi / period)
            b1 = 2.0 * a1 * math.cos(math.sqrt(2.0) * math.pi / period)
            c2 = b1
            c3 = -(a1 * a1)
            c1 = 1.0 - c2 - c3
            us_prev = None
            us_prev2 = None
            p_prev = None
            for i in range(n):
                p = vals[i]
                if p is None:
                    continue
                if us_prev is None or us_prev2 is None or p_prev is None:
                    us = p
                else:
                    us = c1 * (p + p_prev) * 0.5 + c2 * us_prev + c3 * us_prev2
                out[i] = us
                us_prev2 = us_prev
                us_prev = us
                p_prev = p
            return out

        if ind_type == "MaxDrawdown":
            for i in range(n):
                lo = max(0, i - w + 1)
                window_prices = [vals[j] for j in range(lo, i + 1) if vals[j] is not None]
                if len(window_prices) < 2:
                    continue
                peak = -float("inf")
                max_dd = 0.0
                for p in window_prices:
                    assert p is not None
                    peak = max(peak, p)
                    if peak > 0:
                        dd = (p / peak) - 1.0
                        max_dd = min(max_dd, dd)
                out[i] = max_dd * 100.0
            return out

        if ind_type == "Drawdown":
            peak = -float("inf")
            for i in range(n):
                p = vals[i]
                if p is None:
                    continue
                peak = max(peak, p)
                if peak > 0:
                    out[i] = ((p / peak) - 1.0) * 100.0
            return out

        if ind_type in {"AroonUp", "AroonDown", "Aroon"}:
            # QM uses daily highs/lows; cache only has adj close, so close-based approximation.
            p = max(2, w)
            for i in range(n):
                lo = max(0, i - p + 1)
                ws = vals[lo : i + 1]
                pairs = [(j, v) for j, v in enumerate(ws) if v is not None]
                if not pairs:
                    continue
                # Using close as proxy for both high and low.
                hi_idx, _ = max(pairs, key=lambda x: x[1])
                lo_idx, _ = min(pairs, key=lambda x: x[1])
                days_since_hi = (len(ws) - 1) - hi_idx
                days_since_lo = (len(ws) - 1) - lo_idx
                up = ((p - days_since_hi) / p) * 100.0
                dn = ((p - days_since_lo) / p) * 100.0
                if ind_type == "AroonUp":
                    out[i] = up
                elif ind_type == "AroonDown":
                    out[i] = dn
                else:
                    out[i] = up - dn
            return out

        if ind_type in {"Macd", "PercentagePriceOscillator"}:
            ema12: List[Optional[float]] = [None] * n
            ema26: List[Optional[float]] = [None] * n
            alpha12 = 2.0 / (12.0 + 1.0)
            alpha26 = 2.0 / (26.0 + 1.0)
            e12: Optional[float] = None
            e26: Optional[float] = None
            for i in range(n):
                p = vals[i]
                if p is None:
                    continue
                e12 = p if e12 is None else (1 - alpha12) * e12 + alpha12 * p
                e26 = p if e26 is None else (1 - alpha26) * e26 + alpha26 * p
                ema12[i] = e12
                ema26[i] = e26

            core: List[Optional[float]] = [None] * n
            for i in range(n):
                if ema12[i] is None or ema26[i] is None:
                    continue
                if ind_type == "Macd":
                    core[i] = ema12[i] - ema26[i]
                else:
                    if ema26[i] == 0:
                        continue
                    core[i] = ((ema12[i] - ema26[i]) / ema26[i]) * 100.0

            signal: List[Optional[float]] = [None] * n
            alpha9 = 2.0 / (9.0 + 1.0)
            s9: Optional[float] = None
            for i in range(n):
                c = core[i]
                if c is None:
                    continue
                s9 = c if s9 is None else (1 - alpha9) * s9 + alpha9 * c
                signal[i] = s9

            for i in range(n):
                c = core[i]
                s9v = signal[i]
                if c is not None and s9v is not None:
                    out[i] = c - s9v
            return out

        if ind_type == "TrendClarity":
            for i in range(n):
                lo = max(0, i - w + 1)
                ys = [vals[j] for j in range(lo, i + 1)]
                if any(v is None for v in ys):
                    continue
                pts = [float(v) for v in ys if v is not None]
                if len(pts) < 3:
                    continue
                x = list(range(len(pts)))
                xbar = sum(x) / len(x)
                ybar = sum(pts) / len(pts)
                ss_xx = sum((xi - xbar) ** 2 for xi in x)
                if ss_xx == 0:
                    continue
                ss_xy = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, pts))
                b1 = ss_xy / ss_xx
                b0 = ybar - b1 * xbar
                yhat = [b0 + b1 * xi for xi in x]
                ss_res = sum((yi - yh) ** 2 for yi, yh in zip(pts, yhat))
                ss_tot = sum((yi - ybar) ** 2 for yi in pts)
                if ss_tot == 0:
                    out[i] = 100.0
                else:
                    r2 = 1.0 - (ss_res / ss_tot)
                    out[i] = clamp(r2, 0.0, 1.0) * 100.0
            return out

        # Unknown indicator -> None series.
        return out


# ----------------------------- QM strategy compiler -----------------------------


@dataclass
class CompileResult:
    node: Dict[str, Any]
    warnings: List[str]
    symbols: List[str]


class QMCompiler:
    def __init__(self) -> None:
        self.node_id = 0
        self.warnings: List[str] = []
        self.symbols: set[str] = set()

    def _next_id(self) -> int:
        self.node_id += 1
        return self.node_id

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")

    def _normalize_symbol(self, raw: Any, where: str, allow_empty: bool = False) -> str:
        if raw is None:
            if not allow_empty:
                self.warn(where, "missing symbol, using CASHX")
            return "CASHX"
        s = str(raw).strip()
        if not s:
            if not allow_empty:
                self.warn(where, "empty symbol, using CASHX")
            return "CASHX"
        su = s.upper()

        # QuantMage condition pseudo-symbols like `Subspell "Else"` are logic references,
        # not tradable market symbols. Keep them out of symbol coverage/start constraints.
        if su.startswith("SUBSPELL"):
            self.warn(where, f"subspell reference '{s}' is not a market symbol; using CASHX approximation")
            return "CASHX"

        self.symbols.add(su)
        return su

    def compile_condition(self, cond: Any, where: str) -> Dict[str, Any]:
        if not isinstance(cond, dict):
            self.warn(where, "invalid condition; replaced with false")
            return {"condition_type": "AlwaysFalse"}

        ctype = cond.get("condition_type", "SingleCondition")
        if ctype in {"AllOf", "AnyOf"}:
            out = {
                "condition_type": ctype,
                "conditions": [self.compile_condition(c, f"{where}.conditions[{i}]") for i, c in enumerate(cond.get("conditions") or [])],
            }
            return out

        # SingleCondition + sub-types
        lhs = cond.get("lh_indicator") if isinstance(cond.get("lh_indicator"), dict) else {"type": "CurrentPrice", "window": 0}
        rhs = cond.get("rh_indicator") if isinstance(cond.get("rh_indicator"), dict) else {"type": "CurrentPrice", "window": 0}
        return {
            "condition_type": "SingleCondition",
            "type": cond.get("type", "IndicatorAndNumber"),
            "greater_than": bool(cond.get("greater_than", True)),
            "for_days": max(1, safe_int(cond.get("for_days"), 1)),
            "lh_days_ago": max(0, safe_int(cond.get("lh_days_ago"), 0)),
            "rh_days_ago": max(0, safe_int(cond.get("rh_days_ago"), 0)),
            "lh_indicator": {"type": lhs.get("type", "CurrentPrice"), "window": max(0, safe_int(lhs.get("window"), 0))},
            "rh_indicator": {"type": rhs.get("type", "CurrentPrice"), "window": max(0, safe_int(rhs.get("window"), 0))},
            "lh_ticker_symbol": self._normalize_symbol(cond.get("lh_ticker_symbol"), f"{where}.lh_ticker_symbol", allow_empty=True),
            "rh_ticker_symbol": self._normalize_symbol(cond.get("rh_ticker_symbol"), f"{where}.rh_ticker_symbol", allow_empty=True),
            "rh_value": safe_float(cond.get("rh_value"), 0.0),
            "rh_value_int": safe_int(cond.get("rh_value_int"), 0),
            "rh_bias": safe_float(cond.get("rh_bias"), 0.0),
            "rh_weight": safe_float(cond.get("rh_weight"), 1.0),
        }

    def compile_incantation(self, inc: Any, where: str) -> Dict[str, Any]:
        if inc is None:
            return {"node_type": "Cash", "node_id": self._next_id()}
        if not isinstance(inc, dict):
            self.warn(where, "invalid incantation replaced with cash")
            return {"node_type": "Cash", "node_id": self._next_id()}

        itype = inc.get("incantation_type")
        nid = self._next_id()

        if itype == "Ticker":
            return {
                "node_type": "Ticker",
                "node_id": nid,
                "name": str(inc.get("name") or "").strip(),
                "symbol": self._normalize_symbol(inc.get("symbol"), f"{where}.symbol"),
            }

        if itype == "Weighted":
            children = [self.compile_incantation(c, f"{where}.incantations[{i}]") for i, c in enumerate(inc.get("incantations") or [])]
            return {
                "node_type": "Weighted",
                "node_id": nid,
                "name": str(inc.get("name") or "").strip(),
                "weight_type": str(inc.get("type") or "Equal"),
                "weights": [safe_float(x, 0.0) for x in (inc.get("weights") or [])],
                "inverse_volatility_window": max(1, safe_int(inc.get("inverse_volatility_window"), 20)),
                "children": children,
            }

        if itype == "Filtered":
            children = [self.compile_incantation(c, f"{where}.incantations[{i}]") for i, c in enumerate(inc.get("incantations") or [])]
            si = inc.get("sort_indicator") if isinstance(inc.get("sort_indicator"), dict) else {"type": "CumulativeReturn", "window": 20}
            return {
                "node_type": "Filtered",
                "node_id": nid,
                "name": str(inc.get("name") or "").strip(),
                "bottom": bool(inc.get("bottom", False)),
                "count": max(1, safe_int(inc.get("count"), 1)),
                "weight_type": str(inc.get("weight_type") or "Equal"),
                "weights": [safe_float(x, 0.0) for x in (inc.get("weights") or [])],
                "inverse_volatility_window": max(1, safe_int(inc.get("inverse_volatility_window"), 20)),
                "sort_indicator": {"type": si.get("type", "CumulativeReturn"), "window": max(0, safe_int(si.get("window"), 0))},
                "children": children,
            }

        if itype == "IfElse":
            return {
                "node_type": "IfElse",
                "node_id": nid,
                "name": str(inc.get("name") or "").strip(),
                "condition": self.compile_condition(inc.get("condition"), f"{where}.condition"),
                "then": self.compile_incantation(inc.get("then_incantation"), f"{where}.then_incantation"),
                "else": self.compile_incantation(inc.get("else_incantation"), f"{where}.else_incantation"),
            }

        if itype == "Switch":
            return {
                "node_type": "Switch",
                "node_id": nid,
                "name": str(inc.get("name") or "").strip(),
                "conditions": [self.compile_condition(c, f"{where}.conditions[{i}]") for i, c in enumerate(inc.get("conditions") or [])],
                "branches": [self.compile_incantation(c, f"{where}.incantations[{i}]") for i, c in enumerate(inc.get("incantations") or [])],
            }

        if itype == "EnterExit":
            return {
                "node_type": "EnterExit",
                "node_id": nid,
                "name": str(inc.get("name") or "").strip(),
                "enter_condition": self.compile_condition(inc.get("enter_condition"), f"{where}.enter_condition"),
                "exit_condition": self.compile_condition(inc.get("exit_condition"), f"{where}.exit_condition"),
                "enter": self.compile_incantation(inc.get("enter_incantation"), f"{where}.enter_incantation"),
                "exit": self.compile_incantation(inc.get("exit_incantation"), f"{where}.exit_incantation"),
            }

        if itype == "Mixed":
            ind = inc.get("indicator") if isinstance(inc.get("indicator"), dict) else {"type": "RelativeStrengthIndex", "window": 14}
            ts = self._normalize_symbol(inc.get("ticker_symbol"), f"{where}.ticker_symbol")
            return {
                "node_type": "Mixed",
                "node_id": nid,
                "name": str(inc.get("name") or "").strip(),
                "indicator": {"type": ind.get("type", "RelativeStrengthIndex"), "window": max(0, safe_int(ind.get("window"), 0))},
                "ticker_symbol": ts,
                "from_value": safe_float(inc.get("from_value"), 0.0),
                "to_value": safe_float(inc.get("to_value"), 0.0),
                "from": self.compile_incantation(inc.get("from_incantation"), f"{where}.from_incantation"),
                "to": self.compile_incantation(inc.get("to_incantation"), f"{where}.to_incantation"),
            }

        self.warn(where, f"unsupported incantation_type '{itype}', replaced with cash")
        return {"node_type": "Cash", "node_id": nid}

    def compile_strategy(self, qm: Dict[str, Any]) -> CompileResult:
        root = self.compile_incantation(qm.get("incantation"), "incantation")
        return CompileResult(node=root, warnings=self.warnings.copy(), symbols=sorted(self.symbols))


# ----------------------------- Backtest engine -----------------------------


@dataclass
class BacktestOutput:
    strategy_name: str
    start_date: Date
    end_date: Date
    equity_curve: List[Tuple[Date, float]]
    daily_returns: List[Tuple[Date, float]]
    allocations: List[Tuple[Date, Dict[str, float]]]
    benchmark_curve: List[Tuple[Date, float]]
    chart_benchmark_name: str
    chart_benchmark_curve: List[Tuple[Date, float]]
    metrics: Dict[str, float]
    warnings: List[str]
    used_symbols: List[str]
    symbol_coverage: List[Dict[str, Any]]
    limiting_symbols: List[str]
    raw_start_date: Date
    warmup_days: int
    data_source: Dict[str, Any]


class QMBacktester:
    def __init__(self, store: DataStore, strict: bool = False):
        self.store = store
        self.ind = IndicatorEngine(store)
        self.strict = strict
        self.runtime_warnings: List[str] = []
        self._missing_symbol_warned: set[str] = set()
        self._proxied_symbol_warned: set[str] = set()

    def warn(self, msg: str) -> None:
        self.runtime_warnings.append(msg)

    def _portfolio_return(self, alloc: Dict[str, float], idx: int) -> float:
        if not alloc:
            return 0.0
        total = 0.0
        for sym, w in alloc.items():
            r = self.store.daily_return(sym, idx)
            if r is None:
                continue
            total += w * r
        return total

    def _drift_allocation(self, alloc: Dict[str, float], idx: int) -> Dict[str, float]:
        if not alloc:
            return {}
        grown: Dict[str, float] = {}
        for sym, w in alloc.items():
            r = self.store.daily_return(sym, idx)
            if r is None:
                grown[sym] = grown.get(sym, 0.0) + w
            else:
                grown[sym] = grown.get(sym, 0.0) + w * (1.0 + r)
        return self._normalize_alloc(grown)

    def _normalize_alloc(self, alloc: Dict[str, float]) -> Dict[str, float]:
        cleaned = {k: max(0.0, float(v)) for k, v in alloc.items() if v and v > 0}
        s = sum(cleaned.values())
        if s <= 0:
            return {}
        if s > 1.0:
            cleaned = {k: v / s for k, v in cleaned.items()}
        return cleaned

    def _combine_weighted(self, pieces: List[Tuple[Optional[Dict[str, float]], float]]) -> Optional[Dict[str, float]]:
        active = [(alloc, w) for alloc, w in pieces if alloc is not None and w > 0]
        total_w = sum(w for _alloc, w in active)
        if total_w <= 0:
            return None

        out: Dict[str, float] = {}
        for alloc, w in active:
            assert alloc is not None
            nw = w / total_w
            for sym, aw in alloc.items():
                out[sym] = out.get(sym, 0.0) + aw * nw
        return self._normalize_alloc(out)

    def _indicator_from_expr(self, expr: Dict[str, Any], symbol: str, idx: int) -> Optional[float]:
        ind_type = str(expr.get("type", "CurrentPrice"))
        window = max(0, safe_int(expr.get("window"), 0))
        return self.ind.indicator_value(ind_type, symbol, window, idx)

    def _percentile_rank(self, arr: List[Optional[float]], idx: int, lookback: int) -> Optional[float]:
        if idx < 0:
            return None
        cur = arr[idx]
        if cur is None:
            return None
        lo = max(0, idx - lookback + 1)
        ws = [v for v in arr[lo : idx + 1] if v is not None]
        if not ws:
            return None
        le = sum(1 for v in ws if v <= cur)
        return (le / len(ws)) * 100.0

    def _stochastic_rank(self, arr: List[Optional[float]], idx: int, lookback: int) -> Optional[float]:
        if idx < 0:
            return None
        cur = arr[idx]
        if cur is None:
            return None
        lo = max(0, idx - lookback + 1)
        ws = [v for v in arr[lo : idx + 1] if v is not None]
        if not ws:
            return None
        mn, mx = min(ws), max(ws)
        if mx == mn:
            return 50.0
        return ((cur - mn) / (mx - mn)) * 100.0

    def _eval_single_condition(self, cond: Dict[str, Any], idx: int) -> bool:
        kind = str(cond.get("type", "IndicatorAndNumber"))
        gt = bool(cond.get("greater_than", True))
        for_days = max(1, safe_int(cond.get("for_days"), 1))

        def raw_eval(at_idx: int) -> bool:
            if at_idx < 0:
                return False
            lhs_idx = at_idx - max(0, safe_int(cond.get("lh_days_ago"), 0))
            rhs_idx = at_idx - max(0, safe_int(cond.get("rh_days_ago"), 0))

            lhs_expr = cond.get("lh_indicator") or {"type": "CurrentPrice", "window": 0}
            rhs_expr = cond.get("rh_indicator") or {"type": "CurrentPrice", "window": 0}
            lhs_sym = str(cond.get("lh_ticker_symbol") or "CASHX")
            rhs_sym = str(cond.get("rh_ticker_symbol") or "CASHX")

            lhs = self._indicator_from_expr(lhs_expr, lhs_sym, lhs_idx)
            rhs: Optional[float]

            if kind == "BothIndicators":
                base_rhs = self._indicator_from_expr(rhs_expr, rhs_sym, rhs_idx)
                if base_rhs is None:
                    return False
                rhs = base_rhs * safe_float(cond.get("rh_weight"), 1.0) + safe_float(cond.get("rh_bias"), 0.0)

            elif kind == "Month":
                rhs = float(safe_int(cond.get("rh_value_int"), 1))
                lhs = float(self.store.market_days[at_idx].month)

            elif kind in {"IndicatorPercentile", "IndicatorStochastic"}:
                lookback = max(2, safe_int(cond.get("rh_value_int"), 100))
                arr = self.ind.indicator_array(str(lhs_expr.get("type", "CurrentPrice")), lhs_sym, max(0, safe_int(lhs_expr.get("window"), 0)))
                if kind == "IndicatorPercentile":
                    lhs = self._percentile_rank(arr, lhs_idx, lookback)
                else:
                    lhs = self._stochastic_rank(arr, lhs_idx, lookback)
                rhs = safe_float(cond.get("rh_value"), 0.0)

            else:
                # IndicatorAndNumber default
                rhs = safe_float(cond.get("rh_value"), 0.0)

            if lhs is None or rhs is None:
                return False
            return lhs > rhs if gt else lhs < rhs

        for back in range(for_days):
            if not raw_eval(idx - back):
                return False
        return True

    def _eval_condition(self, cond: Dict[str, Any], idx: int) -> bool:
        ctype = cond.get("condition_type")
        if ctype == "AllOf":
            return all(self._eval_condition(c, idx) for c in cond.get("conditions") or [])
        if ctype == "AnyOf":
            return any(self._eval_condition(c, idx) for c in cond.get("conditions") or [])
        if ctype == "SingleCondition":
            return self._eval_single_condition(cond, idx)
        if ctype == "AlwaysFalse":
            return False
        return False

    def _child_volatility(self, alloc: Optional[Dict[str, float]], idx: int, window: int) -> float:
        if not alloc:
            return 0.0
        lo = max(1, idx - max(2, window) + 1)
        rs: List[float] = []
        for d in range(lo, idx + 1):
            r = 0.0
            used = False
            for sym, w in alloc.items():
                dr = self.store.daily_return(sym, d)
                if dr is None:
                    continue
                used = True
                r += w * dr
            if used:
                rs.append(r)
        if len(rs) < 2:
            return 0.0
        return statistics.pstdev(rs) * 100.0

    def _eval_node(self, node: Dict[str, Any], idx: int, state: Dict[int, bool]) -> Optional[Dict[str, float]]:
        t = node.get("node_type")

        if t == "Cash":
            return None

        if t == "Ticker":
            sym = str(node.get("symbol") or "CASHX")
            loaded = self.store.load_symbol(sym)
            if loaded is None:
                if sym not in self._missing_symbol_warned:
                    self._missing_symbol_warned.add(sym)
                    self.warn(f"Missing symbol '{sym}' in cache; treated as cash.")
                return None
            proxied_to = self.store.proxied_symbols.get(sym)
            if proxied_to and sym not in self._proxied_symbol_warned:
                self._proxied_symbol_warned.add(sym)
                self.warn(f"Symbol '{sym}' proxied to '{proxied_to}' (cache fallback).")
            return {sym: 1.0}

        if t == "IfElse":
            cond = node.get("condition") or {}
            branch = node.get("then") if self._eval_condition(cond, idx) else node.get("else")
            return self._eval_node(branch, idx, state)

        if t == "Switch":
            conds = node.get("conditions") or []
            branches = node.get("branches") or []
            n = len(conds)
            c_true = sum(1 for c in conds if self._eval_condition(c, idx))
            # Descending order: branch[0] for all true, ... branch[n] for none true.
            want = n - c_true
            pick = None
            for j in range(want, len(branches)):
                b = branches[j]
                if b is not None:
                    pick = b
                    break
            if pick is None and branches:
                pick = branches[-1]
            if pick is None:
                return None
            return self._eval_node(pick, idx, state)

        if t == "EnterExit":
            nid = int(node.get("node_id"))
            in_pos = bool(state.get(nid, False))
            enter_cond = node.get("enter_condition") or {}
            exit_cond = node.get("exit_condition") or {}
            if in_pos:
                if self._eval_condition(exit_cond, idx):
                    in_pos = False
            else:
                if self._eval_condition(enter_cond, idx):
                    in_pos = True
            state[nid] = in_pos
            return self._eval_node(node.get("enter") if in_pos else node.get("exit"), idx, state)

        if t == "Mixed":
            ind = node.get("indicator") or {"type": "RelativeStrengthIndex", "window": 14}
            symbol = str(node.get("ticker_symbol") or "CASHX")
            x = self._indicator_from_expr(ind, symbol, idx)
            if x is None:
                return self._eval_node(node.get("to"), idx, state)

            fv = safe_float(node.get("from_value"), 0.0)
            tv = safe_float(node.get("to_value"), 100.0)
            if fv == tv:
                w_to = 1.0 if x <= tv else 0.0
            else:
                # At fv => 100% "from", at tv => 100% "to".
                w_to = clamp((x - fv) / (tv - fv), 0.0, 1.0)
            w_from = 1.0 - w_to

            a_from = self._eval_node(node.get("from"), idx, state)
            a_to = self._eval_node(node.get("to"), idx, state)
            return self._combine_weighted([(a_from, w_from), (a_to, w_to)])

        if t == "Weighted":
            children = node.get("children") or []
            wt_type = str(node.get("weight_type") or "Equal")

            child_allocs: List[Tuple[Optional[Dict[str, float]], Dict[str, Any]]] = []
            for c in children:
                ca = self._eval_node(c, idx, state)
                child_allocs.append((ca, c))

            if wt_type == "Equal":
                valid = [(a, c) for a, c in child_allocs if a is not None]
                if not valid:
                    return None
                w = 1.0 / len(valid)
                return self._combine_weighted([(a, w) for a, _ in valid])

            if wt_type == "Custom":
                ws = [safe_float(x, 0.0) for x in (node.get("weights") or [])]
                if not ws:
                    return None
                parts: List[Tuple[Optional[Dict[str, float]], float]] = []
                for i, (a, _c) in enumerate(child_allocs):
                    if i >= len(ws):
                        break
                    parts.append((a, max(0.0, ws[i])))
                return self._combine_weighted(parts)

            # InverseVolatility
            win = max(2, safe_int(node.get("inverse_volatility_window"), 20))
            parts_iv: List[Tuple[Optional[Dict[str, float]], float]] = []
            for a, c in child_allocs:
                if a is None:
                    continue
                vol = self._child_volatility(a, idx, win)
                if vol <= 0:
                    continue
                parts_iv.append((a, 1.0 / vol))
            return self._combine_weighted(parts_iv)

        if t == "Filtered":
            children = node.get("children") or []
            count = max(1, safe_int(node.get("count"), 1))
            bottom = bool(node.get("bottom", False))
            sort_ind = node.get("sort_indicator") or {"type": "CumulativeReturn", "window": 20}

            scored: List[Tuple[float, Dict[str, float], Dict[str, Any]]] = []
            for c in children:
                a = self._eval_node(c, idx, state)
                if a is None:
                    continue
                if c.get("node_type") == "Ticker":
                    sym = str(c.get("symbol") or "CASHX")
                    score = self._indicator_from_expr(sort_ind, sym, idx)
                else:
                    # Weighted average indicator over currently selected child exposure.
                    score = None
                    if a:
                        num = 0.0
                        den = 0.0
                        for sym, w in a.items():
                            sv = self._indicator_from_expr(sort_ind, sym, idx)
                            if sv is None:
                                continue
                            num += w * sv
                            den += w
                        if den > 0:
                            score = num / den
                if score is None:
                    score = -float("inf") if not bottom else float("inf")
                scored.append((float(score), a, c))

            scored.sort(key=lambda x: x[0], reverse=not bottom)
            selected = scored[:count]

            wt_type = str(node.get("weight_type") or "Equal")
            if wt_type == "Equal":
                valid = [(a, c) for _, a, c in selected]
                if not valid:
                    return None
                w = 1.0 / len(valid)
                return self._combine_weighted([(a, w) for a, _ in valid])

            if wt_type == "Custom":
                ws = [max(0.0, safe_float(x, 0.0)) for x in (node.get("weights") or [])]
                if not ws:
                    return None
                parts: List[Tuple[Optional[Dict[str, float]], float]] = []
                for i, (_score, a, _c) in enumerate(selected):
                    if i >= len(ws):
                        break
                    parts.append((a, ws[i]))
                return self._combine_weighted(parts)

            # InverseVolatility weighting for selected.
            win = max(2, safe_int(node.get("inverse_volatility_window"), 20))
            parts_iv: List[Tuple[Optional[Dict[str, float]], float]] = []
            for _score, a, c in selected:
                vol = self._child_volatility(a, idx, win)
                if vol <= 0:
                    continue
                parts_iv.append((a, 1.0 / vol))
            return self._combine_weighted(parts_iv)

        self.warn(f"Unknown node_type '{t}', treated as cash.")
        return None

    def _determine_backtest_start(self, symbols: List[str]) -> Tuple[int, int, List[Dict[str, Any]], List[str]]:
        first_rows: List[Tuple[str, int]] = []
        coverage: List[Dict[str, Any]] = []
        for sym in symbols:
            s = self.store.load_symbol(sym)
            proxied_to = self.store.proxied_symbols.get(sym)
            if s is None:
                coverage.append(
                    {
                        "symbol": sym,
                        "first_date": None,
                        "last_date": None,
                        "proxied_to": proxied_to,
                        "missing": True,
                    }
                )
                continue
            fi = s.first_index()
            li = s.last_index()
            fd = self.store.market_days[fi] if fi is not None else None
            ld = self.store.market_days[li] if li is not None else None
            if fi is not None:
                first_rows.append((sym, fi))
            coverage.append(
                {
                    "symbol": sym,
                    "first_date": fd.isoformat() if fd is not None else None,
                    "last_date": ld.isoformat() if ld is not None else None,
                    "proxied_to": proxied_to,
                    "missing": fi is None,
                }
            )

        if not first_rows:
            return 0, 0, coverage, []

        raw_start = max(idx for _sym, idx in first_rows)
        limiting_symbols = sorted(sym for sym, idx in first_rows if idx == raw_start)
        # Use raw symbol-constrained start; indicators compute with available history.
        return raw_start, raw_start, coverage, limiting_symbols

    @staticmethod
    def _compute_metrics(equity_curve: List[Tuple[Date, float]], daily_returns: List[Tuple[Date, float]]) -> Dict[str, float]:
        if not equity_curve:
            return {}
        start_v = equity_curve[0][1]
        end_v = equity_curve[-1][1]
        n_days = max(1, len(daily_returns))
        years = n_days / 252.0
        cagr = (end_v / start_v) ** (1.0 / years) - 1.0 if years > 0 else 0.0

        peak = -float("inf")
        max_dd = 0.0
        for _d, v in equity_curve:
            peak = max(peak, v)
            if peak > 0:
                dd = (v / peak) - 1.0
                max_dd = min(max_dd, dd)

        rs = [r for _d, r in daily_returns]
        vol = statistics.pstdev(rs) * math.sqrt(252.0) if len(rs) >= 2 else 0.0
        mean = statistics.fmean(rs) if rs else 0.0
        sharpe = (mean * math.sqrt(252.0) / statistics.pstdev(rs)) if len(rs) >= 2 and statistics.pstdev(rs) > 0 else 0.0

        return {
            "total_return": end_v / start_v - 1.0,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "volatility": vol,
            "sharpe": sharpe,
            "start_value": start_v,
            "end_value": end_v,
            "days": float(n_days),
        }

    def run(
        self,
        qm: Dict[str, Any],
        benchmark: Optional[str] = None,
        chart_benchmark: Optional[str] = None,
    ) -> BacktestOutput:
        compiler = QMCompiler()
        cres = compiler.compile_strategy(qm)

        if self.strict and cres.warnings:
            raise ValueError("Strict mode: compiler warnings found:\n" + "\n".join(cres.warnings[:40]))

        used_symbols = set(cres.symbols)
        if benchmark:
            used_symbols.add(benchmark.upper())
        else:
            benchmark = str(qm.get("benchmark_ticker") or "SPY").upper()
            used_symbols.add(benchmark)
        chart_benchmark = str(chart_benchmark or "").upper().strip()
        if chart_benchmark == benchmark:
            chart_benchmark = ""
        if chart_benchmark:
            used_symbols.add(chart_benchmark)

        start_idx, raw_start_idx, symbol_coverage, limiting_symbols = self._determine_backtest_start(sorted(used_symbols))
        end_idx = len(self.store.market_days) - 1
        if start_idx >= end_idx:
            raise ValueError("Not enough history to run backtest.")

        warmup_days = max(0, start_idx - raw_start_idx)
        raw_start_date = self.store.market_days[raw_start_idx]
        effective_start_date = self.store.market_days[start_idx]
        data_source = self.store.data_source_summary()
        source_type = str(data_source.get("type") or "unknown")
        self.warn(
            f"Data source: {source_type}"
            f" (config={data_source.get('config_dir')}; "
            f"tickers={data_source.get('tickers_files')}, "
            f"ext_letf={data_source.get('ext_letf_files')}, "
            f"prices={data_source.get('prices_files')}; "
            f"testfolio_api_enabled={'yes' if data_source.get('testfolio_api_enabled') else 'no'}; "
            f"auth_supplied={'yes' if data_source.get('testfolio_auth_supplied') else 'no'}; "
            f"api_calls={data_source.get('testfolio_api_calls') or 0}; "
            f"api_failures={data_source.get('testfolio_api_failures') or 0})."
        )
        if data_source.get("testfolio_auth_required"):
            reason = str(data_source.get("testfolio_disabled_reason") or "login required")
            self.warn(
                "Testfolio Tactical API requires login or is blocked in this environment; "
                f"falling back to local cache + Yahoo only. ({reason})"
            )
        if limiting_symbols:
            lim_txt = ", ".join(limiting_symbols)
            self.warn(
                "Backtest start constrained by latest first-available symbol(s): "
                f"{lim_txt} at {raw_start_date.isoformat()}, then +{warmup_days} trading-day warmup "
                f"-> effective start {effective_start_date.isoformat()}."
            )
        missing_symbols = sorted(c["symbol"] for c in symbol_coverage if c.get("missing"))
        if missing_symbols:
            self.warn(
                "Symbols missing from local cache (do not contribute to start-date constraint): "
                + ", ".join(missing_symbols)
            )

        # Rebalance configuration
        trading_type = str(qm.get("trading_type") or "Threshold")
        threshold = max(0.0, safe_float(qm.get("threshold"), 0.0)) / 100.0

        state: Dict[int, bool] = {}
        alloc: Dict[str, float] = {}
        equity = 1.0
        bench_equity = 1.0
        chart_bench_equity = 1.0

        equity_curve: List[Tuple[Date, float]] = []
        bench_curve: List[Tuple[Date, float]] = []
        chart_bench_curve: List[Tuple[Date, float]] = []
        daily_returns: List[Tuple[Date, float]] = []
        alloc_rows: List[Tuple[Date, Dict[str, float]]] = []

        # Warm up stateful selector nodes (e.g., EnterExit) before the scoring window.
        for widx in range(0, start_idx):
            self._eval_node(cres.node, widx, state)

        for idx in range(start_idx, end_idx + 1):
            day = self.store.market_days[idx]

            if idx > start_idx:
                pr = self._portfolio_return(alloc, idx)
                br = self.store.daily_return(benchmark, idx) or 0.0
                cbr = (self.store.daily_return(chart_benchmark, idx) or 0.0) if chart_benchmark else 0.0
                equity *= (1.0 + pr)
                bench_equity *= (1.0 + br)
                if chart_benchmark:
                    chart_bench_equity *= (1.0 + cbr)
                daily_returns.append((day, pr))
                # End-of-day weight drift before close rebalance check.
                alloc = self._drift_allocation(alloc, idx)

            equity_curve.append((day, equity))
            bench_curve.append((day, bench_equity))
            if chart_benchmark:
                chart_bench_curve.append((day, chart_bench_equity))

            raw_target = self._eval_node(cres.node, idx, state)
            target = self._normalize_alloc(raw_target or {})

            do_rebalance = True
            if trading_type == "Threshold" and alloc:
                drift = 0.0
                all_syms = set(alloc) | set(target)
                for s in all_syms:
                    drift = max(drift, abs(target.get(s, 0.0) - alloc.get(s, 0.0)))
                do_rebalance = drift >= threshold
            elif trading_type == "Weekly":
                do_rebalance = day.weekday() == 4
            elif trading_type == "Monthly":
                next_day = self.store.market_days[idx + 1] if idx + 1 <= end_idx else None
                do_rebalance = next_day is None or next_day.month != day.month
            elif trading_type == "Quarterly":
                next_day = self.store.market_days[idx + 1] if idx + 1 <= end_idx else None
                do_rebalance = next_day is None or (next_day.month - 1) // 3 != (day.month - 1) // 3
            elif trading_type == "Yearly":
                next_day = self.store.market_days[idx + 1] if idx + 1 <= end_idx else None
                do_rebalance = next_day is None or next_day.year != day.year
            else:
                # Daily + Threshold default behavior.
                pass

            if do_rebalance:
                alloc = target

            # QM/PV style allocation history labels holdings by the next market day
            # after close-time rebalance.
            alloc_day = self.store.market_days[idx + 1] if idx < end_idx else day
            if alloc_rows and alloc_rows[-1][0] == alloc_day:
                alloc_rows[-1] = (alloc_day, alloc.copy())
            else:
                alloc_rows.append((alloc_day, alloc.copy()))

        metrics = self._compute_metrics(equity_curve, daily_returns)

        return BacktestOutput(
            strategy_name=str(qm.get("name") or "Quantmage Strategy").strip() or "Quantmage Strategy",
            start_date=self.store.market_days[start_idx],
            end_date=self.store.market_days[end_idx],
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            allocations=alloc_rows,
            benchmark_curve=bench_curve,
            chart_benchmark_name=chart_benchmark,
            chart_benchmark_curve=chart_bench_curve,
            metrics=metrics,
            warnings=cres.warnings + self.runtime_warnings,
            used_symbols=sorted(used_symbols),
            symbol_coverage=symbol_coverage,
            limiting_symbols=limiting_symbols,
            raw_start_date=raw_start_date,
            warmup_days=warmup_days,
            data_source=data_source,
        )


# ----------------------------- Reporting / export -----------------------------


def write_allocation_csv(path: Path, rows: List[Tuple[Date, Dict[str, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Start Date", "Assets", "Weights"])
        for d, alloc in rows:
            items = sorted(alloc.items(), key=lambda x: x[0])
            if items:
                assets = ", ".join(sym for sym, _wt in items)
                weights = ", ".join(f"{wt * 100.0:.6f}%" for _sym, wt in items)
            else:
                assets = "CASHX"
                weights = "100.000000%"
            w.writerow([format_date_mmddyyyy(d), assets, weights])


def write_equity_csv(path: Path, rows: List[Tuple[Date, float]], bench: List[Tuple[Date, float]]) -> None:
    bench_map = {d: v for d, v in bench}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Strategy Equity", "Benchmark Equity"])
        for d, v in rows:
            w.writerow([d.isoformat(), f"{v:.10f}", f"{bench_map.get(d, float('nan')):.10f}"])


def write_symbol_coverage_csv(path: Path, out: BacktestOutput) -> None:
    limiting_set = set(out.limiting_symbols)
    rows = sorted(
        out.symbol_coverage,
        key=lambda r: (r.get("first_date") is None, r.get("first_date") or "", r.get("symbol") or ""),
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Symbol", "First Date", "Last Date", "Proxied To", "Missing", "Limits Start"])
        for r in rows:
            sym = str(r.get("symbol") or "")
            w.writerow(
                [
                    sym,
                    r.get("first_date") or "",
                    r.get("last_date") or "",
                    r.get("proxied_to") or "",
                    "yes" if r.get("missing") else "no",
                    "yes" if sym in limiting_set else "no",
                ]
            )


def build_start_diagnostics_lines(out: BacktestOutput) -> List[str]:
    src = out.data_source or {}
    network_calls = bool(src.get("network_calls"))
    lines = [
        f"effective_start={out.start_date.isoformat()}",
        f"raw_symbol_start={out.raw_start_date.isoformat()}",
        f"warmup_trading_days={out.warmup_days}",
        f"limiting_symbols={','.join(out.limiting_symbols) if out.limiting_symbols else ''}",
        f"data_source_type={src.get('type') or ''}",
        f"data_source_config={src.get('config_dir') or ''}",
        f"data_source_tickers_files={src.get('tickers_files') or 0}",
        f"data_source_ext_letf_files={src.get('ext_letf_files') or 0}",
        f"data_source_prices_files={src.get('prices_files') or 0}",
        f"network_api_calls={'yes' if network_calls else 'no'}",
        f"testfolio_api_enabled={'yes' if src.get('testfolio_api_enabled') else 'no'}",
        f"testfolio_auth_supplied={'yes' if src.get('testfolio_auth_supplied') else 'no'}",
        f"testfolio_api_calls={src.get('testfolio_api_calls') or 0}",
        f"testfolio_api_failures={src.get('testfolio_api_failures') or 0}",
        f"testfolio_auth_required={'yes' if src.get('testfolio_auth_required') else 'no'}",
        f"testfolio_disabled_reason={src.get('testfolio_disabled_reason') or ''}",
        f"yahoo_fallback_enabled={'yes' if src.get('yahoo_fallback_enabled') else 'no'}",
        f"yahoo_api_calls={src.get('yahoo_api_calls') or 0}",
        f"yahoo_api_failures={src.get('yahoo_api_failures') or 0}",
    ]
    return lines


def write_start_diagnostics_txt(path: Path, out: BacktestOutput) -> None:
    lines = build_start_diagnostics_lines(out)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_limiting_symbols_csv(path: Path, out: BacktestOutput) -> None:
    coverage_by_symbol = {str(r.get("symbol") or ""): r for r in out.symbol_coverage}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Symbol", "First Date", "Effective Start", "Warmup Days"])
        for sym in sorted(out.limiting_symbols):
            rec = coverage_by_symbol.get(sym, {})
            w.writerow([sym, rec.get("first_date") or "", out.start_date.isoformat(), out.warmup_days])


def _template_candidates() -> List[Path]:
    base = Path(__file__).resolve().parent
    return [
        base / "reports" / "Standard_Short_logic_5BjE8lUAueEOt9GM1KGn_vs_QQQ.html",
        base / "reports" / "EDIT2_Short_logic_fKwTR9qR4Uuq8vo9NI8L_vs_QQQ.html",
    ]


def _load_rain_template() -> str:
    for p in _template_candidates():
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise FileNotFoundError("Rain HTML template not found in reports/.")


def _extract_metric_meta(template_html: str) -> List[Tuple[str, str, str]]:
    row_re = re.compile(
        r"<tr><td>(?P<label>.*?)<span class=\"info\" tabindex=\"0\" "
        r"data-desc-id=\"metric-(?P<id>[^\"]+)-desc\"[^>]*>ⓘ</span></td><td>.*?</td><td>.*?</td></tr>",
        re.S,
    )
    desc_map: Dict[str, str] = {
        mid: html.unescape(desc.strip())
        for mid, desc in re.findall(r"<div id=\"metric-([^\"]+)-desc\" class=\"sr-only\">(.*?)</div>", template_html, re.S)
    }
    out: List[Tuple[str, str, str]] = []
    for m in row_re.finditer(template_html):
        mid = m.group("id")
        raw_label = m.group("label")
        label = html.unescape(re.sub(r"<[^>]+>", "", raw_label)).strip()
        out.append((mid, label, desc_map.get(mid, "")))
    return out


def _replace_js_const(raw: str, name: str, obj: Any) -> str:
    repl = f"const {name} = {json.dumps(obj, separators=(',', ':'))};\n"
    pat = rf"const {re.escape(name)} = .*?;\n"
    updated, n = re.subn(pat, repl, raw, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"Unable to replace JS const '{name}' in template.")
    return updated


def _insert_js_const_after(raw: str, after_name: str, name: str, obj: Any) -> str:
    repl = f"const {name} = {json.dumps(obj, separators=(',', ':'))};\n"
    pat = rf"(const {re.escape(after_name)} = .*?;\n)"
    updated, n = re.subn(pat, lambda m: m.group(1) + repl, raw, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"Unable to insert JS const '{name}' after '{after_name}' in template.")
    return updated


def _patch_cumulative_chart_for_extra_benchmark(raw: str, extra_benchmark: str) -> str:
    title_old = (
        'Cumulative Returns vs <span class="hint" tabindex="0" data-desc-id="benchmark-name-desc">Benchmark</span>'
        ' <span class="info" tabindex="0" data-desc-id="chart-eqChart-desc">ⓘ</span></h3>'
    )
    title_new = (
        'Cumulative Returns vs <span class="hint" tabindex="0" data-desc-id="benchmark-name-desc">Benchmark</span>'
        f' and {html.escape(extra_benchmark)} <span class="info" tabindex="0" data-desc-id="chart-eqChart-desc">ⓘ</span></h3>'
    )
    raw = raw.replace(title_old, title_new, 1)
    desc_old = "Cumulative returns (equity curve) showing growth of $1 invested; supports comparison between strategy and benchmark."
    desc_new = (
        "Cumulative returns (equity curve) showing growth of $1 invested; supports comparison between strategy, "
        f"benchmark, and {html.escape(extra_benchmark)}."
    )
    raw = raw.replace(desc_old, desc_new, 1)

    raw = raw.replace(
        "  if (benchmarkData) {\n    traces.push({ x: benchmarkData.dates, y: benchmarkData.equity, name: 'Benchmark', mode: 'lines' });\n  }\n",
        "  if (benchmarkData) {\n    traces.push({ x: benchmarkData.dates, y: benchmarkData.equity, name: 'Benchmark', mode: 'lines' });\n  }\n"
        "  if (extraBenchmarkData) {\n    traces.push({ x: extraBenchmarkData.dates, y: extraBenchmarkData.equity, name: extraBenchmarkData.name, mode: 'lines' });\n  }\n",
        1,
    )
    raw = raw.replace(
        "      const allVals = [...strategyData.equity, ...benchmarkData.equity, ...diffEquity];\n",
        "      const allVals = [...strategyData.equity, ...benchmarkData.equity, ...(extraBenchmarkData ? extraBenchmarkData.equity : []), ...diffEquity];\n",
        1,
    )
    raw = raw.replace(
        "      traces.push({\n        x: strategyData.dates,\n        y: asinhTransform(strategyData.equity, scale),\n        customdata: strategyData.equity,\n        name: 'Strategy',\n        mode: 'lines',\n        hovertemplate: '%{customdata:.2f}x<extra>Strategy</extra>'\n      });\n",
        "      if (extraBenchmarkData) {\n"
        "        traces.push({\n"
        "          x: extraBenchmarkData.dates,\n"
        "          y: asinhTransform(extraBenchmarkData.equity, scale),\n"
        "          customdata: extraBenchmarkData.equity,\n"
        "          name: extraBenchmarkData.name,\n"
        "          mode: 'lines',\n"
        "          hovertemplate: '%{customdata:.2f}x<extra>%{fullData.name}</extra>'\n"
        "        });\n"
        "      }\n"
        "      traces.push({\n        x: strategyData.dates,\n        y: asinhTransform(strategyData.equity, scale),\n        customdata: strategyData.equity,\n        name: 'Strategy',\n        mode: 'lines',\n        hovertemplate: '%{customdata:.2f}x<extra>Strategy</extra>'\n      });\n",
        1,
    )
    return raw


def _fmt_long_date(d: Date) -> str:
    return d.strftime("%b %d, %Y").replace(" 0", " ")


def _series_from_curve(curve: List[Tuple[Date, float]]) -> Tuple[List[str], List[float], List[float], List[float]]:
    if not curve:
        return [], [], [], []
    dates = [d.isoformat() for d, _ in curve]
    equity = [float(v) for _, v in curve]
    returns: List[float] = [0.0]
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        cur = equity[i]
        returns.append((cur / prev - 1.0) if prev else 0.0)
    drawdown: List[float] = []
    peak = -float("inf")
    for v in equity:
        peak = max(peak, v)
        drawdown.append((v / peak - 1.0) if peak > 0 else 0.0)
    return dates, equity, returns, drawdown


def _resample_compounded(returns: "pd.Series", rule: str) -> List[float]:
    agg = (1.0 + returns).resample(rule).prod() - 1.0
    return [float(x) for x in agg.dropna().tolist()]


def _build_monthly_heatmap(returns: "pd.Series") -> Dict[str, Any]:
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    if monthly.empty:
        return {"years": [], "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], "values": []}
    years = sorted({int(ts.year) for ts in monthly.index})
    year_to_i = {y: i for i, y in enumerate(years)}
    vals: List[List[Optional[float]]] = [[None for _ in range(12)] for _ in years]
    for ts, v in monthly.items():
        vals[year_to_i[int(ts.year)]][int(ts.month) - 1] = float(v)
    return {
        "years": years,
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "values": vals,
    }


def _build_eoy_data(returns: "pd.Series") -> Dict[str, Any]:
    yearly = (1.0 + returns).resample("YE").prod() - 1.0
    return {
        "years": [int(ts.year) for ts in yearly.index],
        "returns": [float(v) for v in yearly.tolist()],
    }


def _rolling_block(values: List[float], dates_iso: List[str], window: int) -> Dict[str, Any]:
    if window <= 1 or len(values) < window:
        return {"dates": [], "sharpe": [], "sortino": [], "volatility": [], "cagr": []}
    out_dates: List[str] = []
    out_sharpe: List[Optional[float]] = []
    out_sortino: List[Optional[float]] = []
    out_vol: List[Optional[float]] = []
    out_cagr: List[Optional[float]] = []
    for i in range(window - 1, len(values)):
        ws = values[i - window + 1 : i + 1]
        out_dates.append(dates_iso[i])
        mean = statistics.fmean(ws)
        std = statistics.pstdev(ws) if len(ws) >= 2 else 0.0
        sharpe = (mean * math.sqrt(252.0) / std) if std > 0 else None
        downside_sq = [min(0.0, r) ** 2 for r in ws]
        downside_dev = math.sqrt(statistics.fmean(downside_sq)) if downside_sq else 0.0
        sortino = ((mean * 252.0) / (downside_dev * math.sqrt(252.0))) if downside_dev > 0 else None
        vol = std * math.sqrt(252.0) if std > 0 else None
        comp = 1.0
        for r in ws:
            comp *= (1.0 + r)
        cagr = (comp ** (252.0 / float(window)) - 1.0) if comp > 0 else None
        out_sharpe.append(None if sharpe is None else float(sharpe))
        out_sortino.append(None if sortino is None else float(sortino))
        out_vol.append(None if vol is None else float(vol))
        out_cagr.append(None if cagr is None else float(cagr))
    return {"dates": out_dates, "sharpe": out_sharpe, "sortino": out_sortino, "volatility": out_vol, "cagr": out_cagr}


def _build_drawdown_periods(dates_iso: List[str], equity: List[float]) -> Dict[str, Any]:
    if not dates_iso or not equity:
        return {"defaultPctThreshold": 5.0, "defaultDaysThreshold": 126, "periods": []}

    periods: List[Dict[str, Any]] = []
    peak_val = equity[0]
    peak_idx = 0
    in_dd = False
    start_idx = 0
    min_dd = 0.0

    for i, v in enumerate(equity):
        if v >= peak_val:
            if in_dd:
                end_idx = i
                days = max(1, end_idx - start_idx + 1)
                periods.append(
                    {
                        "start": dates_iso[start_idx],
                        "end": dates_iso[end_idx],
                        "pct": round(abs(min_dd) * 100.0, 1),
                        "days": days,
                    }
                )
                in_dd = False
            peak_val = v
            peak_idx = i
            continue
        dd = (v / peak_val - 1.0) if peak_val > 0 else 0.0
        if not in_dd:
            in_dd = True
            start_idx = peak_idx
            min_dd = dd
        else:
            min_dd = min(min_dd, dd)

    if in_dd:
        days = max(1, len(equity) - start_idx)
        periods.append(
            {
                "start": dates_iso[start_idx],
                "end": None,
                "pct": round(abs(min_dd) * 100.0, 1),
                "days": days,
            }
        )

    periods = [p for p in periods if p["pct"] > 0.0]
    periods_by_pct = sorted(periods, key=lambda x: (x["pct"], x["days"]), reverse=True)
    periods_by_days = sorted(periods, key=lambda x: (x["days"], x["pct"]), reverse=True)
    if periods_by_pct:
        pick_pct = periods_by_pct[min(9, len(periods_by_pct) - 1)]["pct"]
        pick_days = periods_by_days[min(9, len(periods_by_days) - 1)]["days"]
    else:
        pick_pct = 5.0
        pick_days = 126
    return {
        "defaultPctThreshold": float(round(max(0.1, float(pick_pct)), 1)),
        "defaultDaysThreshold": int(max(1, int(pick_days))),
        "periods": periods_by_pct[:300],
    }


def _aligned_return_series(out: BacktestOutput) -> Tuple["pd.Series", "pd.Series"]:
    import pandas as pd  # type: ignore

    s_dates, _s_eq, s_ret, _s_dd = _series_from_curve(out.equity_curve)
    b_dates, _b_eq, b_ret, _b_dd = _series_from_curve(out.benchmark_curve)
    s = pd.Series(s_ret, index=pd.to_datetime(s_dates), dtype="float64")
    b = pd.Series(b_ret, index=pd.to_datetime(b_dates), dtype="float64")
    aligned = pd.concat([s, b], axis=1, join="inner").dropna()
    if aligned.empty:
        return s.iloc[0:0], b.iloc[0:0]
    return aligned.iloc[:, 0], aligned.iloc[:, 1]


def _safe_qs(callable_obj: Any, *args: Any, **kwargs: Any) -> Optional[float]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            v = callable_obj(*args, **kwargs)
        if v is None:
            return None
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    except Exception:
        return None


def _compute_metric_bundle(returns: "pd.Series", benchmark: Optional["pd.Series"]) -> Dict[str, Optional[float]]:
    import pandas as pd  # type: ignore
    import quantstats as qs  # type: ignore

    ret = returns.dropna()
    if ret.empty:
        return {}
    rf = 0.0
    n_days = max(1, len(ret))
    years = n_days / 252.0

    total_return = float((1.0 + ret).prod() - 1.0)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    dd_series = qs.stats.to_drawdown_series(ret)
    max_dd = float(dd_series.min()) if not dd_series.empty else 0.0
    dd_details = qs.stats.drawdown_details(dd_series)
    if isinstance(dd_details, pd.DataFrame) and not dd_details.empty:
        dd_days = dd_details["days"].astype(float)
        dd_mag = dd_details["max drawdown"].astype(float) / 100.0
        longest_dd_days = float(dd_days.max())
        deepest_idx = dd_mag.idxmin()
        max_dd_days = float(dd_days.loc[deepest_idx])
        avg_dd = float(dd_mag.mean())
        avg_dd_days = float(dd_days.mean())
    else:
        longest_dd_days = 0.0
        max_dd_days = 0.0
        avg_dd = 0.0
        avg_dd_days = 0.0

    vol_ann = _safe_qs(qs.stats.volatility, ret, periods=252, annualize=True)
    sharpe = _safe_qs(qs.stats.sharpe, ret, rf=rf, periods=252, annualize=True)
    prob_sharpe = _safe_qs(qs.stats.probabilistic_sharpe_ratio, ret, rf=rf, periods=252, annualize=False)
    smart_sharpe = _safe_qs(qs.stats.smart_sharpe, ret, rf=rf, periods=252, annualize=True)
    sortino = _safe_qs(qs.stats.sortino, ret, rf=rf, periods=252, annualize=True)
    smart_sortino = _safe_qs(qs.stats.smart_sortino, ret, rf=rf, periods=252, annualize=True)
    omega = _safe_qs(qs.stats.omega, ret, rf=rf, required_return=0.0, periods=252)
    ulcer_index = _safe_qs(qs.stats.ulcer_index, ret)
    serenity_index = _safe_qs(qs.stats.serenity_index, ret, rf=rf)
    calmar = _safe_qs(qs.stats.calmar, ret)
    skew = _safe_qs(qs.stats.skew, ret)
    kurtosis = _safe_qs(qs.stats.kurtosis, ret)
    expected_daily = _safe_qs(qs.stats.expected_return, ret)
    expected_month = _safe_qs(qs.stats.expected_return, ret, aggregate="ME")
    expected_year = _safe_qs(qs.stats.expected_return, ret, aggregate="YE")
    daily_var = _safe_qs(qs.stats.var, ret)
    expected_shortfall = _safe_qs(qs.stats.expected_shortfall, ret)
    recovery_factor = _safe_qs(qs.stats.recovery_factor, ret, rf=rf)
    profit_factor = _safe_qs(qs.stats.profit_factor, ret)

    monthly = (1.0 + ret).resample("ME").prod() - 1.0
    quarterly = (1.0 + ret).resample("QE").prod() - 1.0
    yearly = (1.0 + ret).resample("YE").prod() - 1.0

    def trailing_comp(series: "pd.Series", start_ts: "pd.Timestamp") -> Optional[float]:
        s = series[series.index >= start_ts]
        if s.empty:
            return None
        return float((1.0 + s).prod() - 1.0)

    def trailing_annualized(series: "pd.Series", start_ts: "pd.Timestamp") -> Optional[float]:
        s = series[series.index >= start_ts]
        if s.empty:
            return None
        comp = float((1.0 + s).prod() - 1.0)
        yrs = max(len(s) / 252.0, 1e-9)
        return float((1.0 + comp) ** (1.0 / yrs) - 1.0)

    end_ts = ret.index[-1]
    mtd = trailing_comp(ret, end_ts.replace(day=1))
    ytd = trailing_comp(ret, end_ts.replace(month=1, day=1))
    three_m = trailing_comp(ret, end_ts - pd.DateOffset(months=3))
    six_m = trailing_comp(ret, end_ts - pd.DateOffset(months=6))
    one_y = trailing_comp(ret, end_ts - pd.DateOffset(years=1))
    three_y = trailing_annualized(ret, end_ts - pd.DateOffset(years=3))
    five_y = trailing_annualized(ret, end_ts - pd.DateOffset(years=5))
    ten_y = trailing_annualized(ret, end_ts - pd.DateOffset(years=10))

    wins = int((ret > 0).sum())
    losses = int((ret < 0).sum())
    samples = max(1, wins + losses)
    win_days_pct = wins / float(samples)
    win_month_pct = float((monthly > 0).sum() / max(1, len(monthly))) if len(monthly) else 0.0
    win_quarter_pct = float((quarterly > 0).sum() / max(1, len(quarterly))) if len(quarterly) else 0.0
    win_year_pct = float((yearly > 0).sum() / max(1, len(yearly))) if len(yearly) else 0.0

    pair_alpha: Optional[float] = None
    pair_beta: Optional[float] = None
    pair_corr: Optional[float] = None
    pair_r2: Optional[float] = None
    pair_ir: Optional[float] = None
    if benchmark is not None and not benchmark.empty:
        aligned = pd.concat([ret, benchmark], axis=1, join="inner").dropna()
        if not aligned.empty:
            sret = aligned.iloc[:, 0]
            bret = aligned.iloc[:, 1]
            greeks = qs.stats.greeks(sret, bret)
            pair_alpha = _safe_qs(lambda: greeks.get("alpha"))
            pair_beta = _safe_qs(lambda: greeks.get("beta"))
            pair_corr = _safe_qs(sret.corr, bret)
            pair_r2 = (pair_corr * pair_corr) if pair_corr is not None else None
            pair_ir = _safe_qs(qs.stats.information_ratio, sret, bret)

    romad = (cagr / abs(max_dd)) if max_dd < 0 else None
    sortino_sqrt2 = (sortino / math.sqrt(2.0)) if sortino is not None else None
    smart_sqrt2 = (smart_sortino / math.sqrt(2.0)) if smart_sortino is not None else None
    risk_free_rate = rf
    rf_total = (1.0 + rf) ** years - 1.0
    excess_return = total_return - rf_total
    excess_cagr = cagr - rf

    return {
        "time_in_market": _safe_qs(qs.stats.exposure, ret),
        "risk_free_rate": risk_free_rate,
        "total_return": total_return,
        "excess_return": excess_return,
        "cagr": cagr,
        "excess_cagr": excess_cagr,
        "sharpe": sharpe,
        "romad": romad,
        "prob_sharpe": prob_sharpe,
        "smart_sharpe": smart_sharpe,
        "sortino": sortino,
        "smart_sortino": smart_sortino,
        "sortino_sqrt2": sortino_sqrt2,
        "smart_sqrt2": smart_sqrt2,
        "omega": omega,
        "ulcer_index": ulcer_index,
        "serenity_index": serenity_index,
        "max_dd": max_dd,
        "max_dd_days": max_dd_days,
        "longest_dd_days": longest_dd_days,
        "avg_dd": avg_dd,
        "avg_dd_days": avg_dd_days,
        "vol_ann": vol_ann,
        "corr": pair_corr,
        "r2": pair_r2,
        "ir": pair_ir,
        "calmar": calmar,
        "skew": skew,
        "kurtosis": kurtosis,
        "expected_daily": expected_daily,
        "expected_month": expected_month,
        "expected_year": expected_year,
        "daily_var": daily_var,
        "expected_shortfall": expected_shortfall,
        "recovery_factor": recovery_factor,
        "mtd": mtd,
        "three_m": three_m,
        "six_m": six_m,
        "ytd": ytd,
        "one_y": one_y,
        "three_y": three_y,
        "five_y": five_y,
        "ten_y": ten_y,
        "all_time": cagr,
        "best_day": _safe_qs(qs.stats.best, ret),
        "worst_day": _safe_qs(qs.stats.worst, ret),
        "best_month": _safe_qs(qs.stats.best, ret, aggregate="ME"),
        "worst_month": _safe_qs(qs.stats.worst, ret, aggregate="ME"),
        "best_year": _safe_qs(qs.stats.best, ret, aggregate="YE"),
        "worst_year": _safe_qs(qs.stats.worst, ret, aggregate="YE"),
        "avg_up_month": _safe_qs(qs.stats.avg_win, ret, aggregate="ME"),
        "avg_down_month": _safe_qs(qs.stats.avg_loss, ret, aggregate="ME"),
        "profit_factor": profit_factor,
        "win_ratio": (wins / float(losses)) if losses > 0 else (float("inf") if wins > 0 else 0.0),
        "win_days": float(wins),
        "loss_days": float(losses),
        "win_days_pct": win_days_pct,
        "win_month_pct": win_month_pct,
        "win_quarter_pct": win_quarter_pct,
        "win_year_pct": win_year_pct,
        "alpha": pair_alpha,
        "beta": pair_beta,
    }


def _build_allocations_data(out: BacktestOutput, benchmark: str) -> Dict[str, Any]:
    dates = [d.isoformat() for d, _ in out.allocations]
    raw_allocs = [a for _, a in out.allocations]
    all_tickers = sorted({sym for alloc in raw_allocs for sym in alloc})
    cash_vals: List[float] = []
    for alloc in raw_allocs:
        s = sum(max(0.0, float(v)) for v in alloc.values())
        cash_vals.append(max(0.0, 1.0 - s))
    if any(v > 1e-8 for v in cash_vals):
        all_tickers.append("CASH")

    weights: Dict[str, List[float]] = {t: [] for t in all_tickers}
    for i, alloc in enumerate(raw_allocs):
        for t in all_tickers:
            if t == "CASH":
                weights[t].append(float(cash_vals[i]))
            else:
                weights[t].append(float(max(0.0, alloc.get(t, 0.0))))

    all_tickers.sort(key=lambda t: statistics.fmean(weights[t]) if weights[t] else 0.0, reverse=True)
    strat_obj = {"dates": dates, "tickers": all_tickers, "weights": {t: weights[t] for t in all_tickers}}
    bench_obj = {
        "dates": dates,
        "tickers": [benchmark],
        "weights": {benchmark: [1.0 for _ in dates]},
    }
    return {"strategy": strat_obj, "benchmark": bench_obj}


def _format_metric_value(metric_id: str, value: Optional[float]) -> str:
    if value is None:
        return "-"
    if math.isnan(value) or math.isinf(value):
        if metric_id == "win_ratio" and value > 0:
            return "∞"
        return "-"

    percent_ids = {
        "time_in_market", "risk_free_rate", "total_return", "excess_return", "cagr", "excess_cagr",
        "prob_sharpe", "max_dd", "avg_dd", "vol_ann", "expected_daily", "expected_month",
        "expected_year", "daily_var", "expected_shortfall", "mtd", "three_m", "six_m",
        "ytd", "one_y", "three_y", "five_y", "ten_y", "all_time", "best_day", "worst_day",
        "best_month", "worst_month", "best_year", "worst_year", "avg_up_month", "avg_down_month",
        "win_days_pct", "win_month_pct", "win_quarter_pct", "win_year_pct",
    }
    int_ids = {"max_dd_days", "longest_dd_days", "avg_dd_days", "win_days", "loss_days"}

    if metric_id in int_ids:
        return f"{int(round(value)):,}"
    if metric_id in percent_ids:
        s = f"{value * 100.0:,.2f}".rstrip("0").rstrip(".")
        return s + "%"
    s = f"{value:,.3f}".rstrip("0").rstrip(".")
    return s


def _render_right_column(
    strategy_name: str,
    benchmark: str,
    metric_meta: List[Tuple[str, str, str]],
    strat_vals: Dict[str, Optional[float]],
    bench_vals: Dict[str, Optional[float]],
) -> str:
    def card_percent(v: Optional[float]) -> str:
        return _format_metric_value("cagr", v)

    def card_num(v: Optional[float]) -> str:
        return "-" if v is None else f"{v:.3f}".rstrip("0").rstrip(".")

    table_rows = []
    for mid, label, desc in metric_meta:
        sv = strat_vals.get(mid)
        if mid in {"corr", "r2", "ir"}:
            bv = sv
        else:
            bv = bench_vals.get(mid)
        s_fmt = _format_metric_value(mid, sv)
        b_fmt = _format_metric_value(mid, bv)
        table_rows.append(
            "<tr><td>"
            + html.escape(label)
            + f" <span class=\"info\" tabindex=\"0\" data-desc-id=\"metric-{mid}-desc\" "
            + f"aria-describedby=\"metric-{mid}-desc\">ⓘ</span></td>"
            + f"<td>{html.escape(s_fmt)}</td><td>{html.escape(b_fmt)}</td></tr>\n"
            + f"<div id=\"metric-{mid}-desc\" class=\"sr-only\">{html.escape(desc)}</div>"
        )

    alpha_val = card_num(strat_vals.get("alpha"))
    beta_val = card_num(strat_vals.get("beta"))
    corr_val = card_num(strat_vals.get("corr"))
    r2_val = card_num(strat_vals.get("r2"))

    return f"""
<div class="right-column">
<div class="metric-main">
  <div class="metric-title">Annual Return<span class="info" tabindex="0" data-description="Compound Annual Growth Rate — the annualized rate of return over the backtest period.">ⓘ</span></div>
  <h1>{html.escape(card_percent(strat_vals.get("cagr")))}</h1>
</div>
<div class="metric-sub-container">
  <div class="metric-sub">
    <div class="metric-title">Max Drawdown <span class="info" tabindex="0" data-description="Largest peak-to-trough decline during the backtest period.">ⓘ</span></div>
    <h2>{html.escape(_format_metric_value("max_dd", strat_vals.get("max_dd")))}</h2>
  </div>
  <div class="metric-sub">
    <div class="metric-title">Longest DD Days <span class="info" tabindex="0" data-description="Duration of the longest drawdown period in trading days.">ⓘ</span></div>
    <h2>{html.escape(_format_metric_value("longest_dd_days", strat_vals.get("longest_dd_days")))}</h2>
  </div>
  <div class="metric-sub">
    <div class="metric-title">RoMaD <span class="info" tabindex="0" data-description="Return over Maximum Drawdown — Annual Return divided by the absolute value of Max Drawdown.">ⓘ</span></div>
    <h2>{html.escape(_format_metric_value("romad", strat_vals.get("romad")))}</h2>
  </div>
  <div class="metric-sub">
    <div class="metric-title">Calmar <span class="info" tabindex="0" data-description="Annualized return divided by maximum drawdown magnitude.">ⓘ</span></div>
    <h2>{html.escape(_format_metric_value("calmar", strat_vals.get("calmar")))}</h2>
  </div>
  <div class="metric-sub">
    <div class="metric-title">Alpha <span class="info" tabindex="0" data-description="Excess return of the strategy compared to the benchmark, adjusted for risk.">ⓘ</span></div>
    <h2>{html.escape(alpha_val)}</h2>
  </div>
  <div class="metric-sub">
    <div class="metric-title">Beta <span class="info" tabindex="0" data-description="Measure of the strategy's volatility relative to the benchmark.">ⓘ</span></div>
    <h2>{html.escape(beta_val)}</h2>
  </div>
  <div class="metric-sub">
    <div class="metric-title">Correlation <span class="info" tabindex="0" data-description="Pearson correlation coefficient between strategy and benchmark daily returns.">ⓘ</span></div>
    <h2>{html.escape(corr_val)}</h2>
  </div>
  <div class="metric-sub">
    <div class="metric-title">R² <span class="info" tabindex="0" data-description="Coefficient of determination between strategy and benchmark daily returns.">ⓘ</span></div>
    <h2>{html.escape(r2_val)}</h2>
  </div>
</div>
  <h3>Key Performance Metrics</h3>
<table>
  <thead>
    <tr><th>Metric</th><th><span class="hint" tabindex="0" data-desc-id="strategy-name-desc">Strategy</span></th><th><span class="hint" tabindex="0" data-desc-id="benchmark-name-desc">Benchmark</span></th></tr>
  </thead>
  <tbody>
    {''.join(table_rows)}
  </tbody>
</table>
</div>
"""


def _write_report_html_rain_style(path: Path, out: BacktestOutput, benchmark: str) -> None:
    import pandas as pd  # type: ignore

    template = _load_rain_template()
    metric_meta = _extract_metric_meta(template)
    if not metric_meta:
        raise ValueError("Could not parse Rain metric schema from template.")

    s_dates, s_eq, s_ret, s_dd = _series_from_curve(out.equity_curve)
    b_dates, b_eq, b_ret, b_dd = _series_from_curve(out.benchmark_curve)
    c_dates, c_eq, c_ret, c_dd = _series_from_curve(out.chart_benchmark_curve)

    s_series = pd.Series(s_ret, index=pd.to_datetime(s_dates), dtype="float64")
    b_series = pd.Series(b_ret, index=pd.to_datetime(b_dates), dtype="float64")

    strategy_data = {
        "dates": s_dates,
        "equity": s_eq,
        "returns": s_ret,
        "drawdown": s_dd,
        "weeklyReturns": _resample_compounded(s_series, "W-FRI"),
        "monthlyReturns": _resample_compounded(s_series, "ME"),
        "quarterlyReturns": _resample_compounded(s_series, "QE"),
    }
    benchmark_data: Optional[Dict[str, Any]] = {
        "dates": b_dates,
        "equity": b_eq,
        "returns": b_ret,
        "drawdown": b_dd,
        "weeklyReturns": _resample_compounded(b_series, "W-FRI"),
        "monthlyReturns": _resample_compounded(b_series, "ME"),
        "quarterlyReturns": _resample_compounded(b_series, "QE"),
    } if b_dates else None
    chart_benchmark_data: Optional[Dict[str, Any]] = {
        "name": out.chart_benchmark_name,
        "dates": c_dates,
        "equity": c_eq,
        "returns": c_ret,
        "drawdown": c_dd,
    } if out.chart_benchmark_name and c_dates else None

    rolling_data = {
        "strategy": {f"w{w}": _rolling_block(s_ret, s_dates, w) for w in [21, 63, 126, 252]},
        "benchmark": ({f"w{w}": _rolling_block(b_ret, b_dates, w) for w in [21, 63, 126, 252]} if benchmark_data else None),
    }

    monthly_heatmap_data = {
        "strategy": _build_monthly_heatmap(s_series),
        "benchmark": _build_monthly_heatmap(b_series) if benchmark_data else None,
    }
    eoy_data = {
        "strategy": _build_eoy_data(s_series),
        "benchmark": _build_eoy_data(b_series) if benchmark_data else None,
    }
    drawdown_periods = {
        "strategy": _build_drawdown_periods(s_dates, s_eq),
        "benchmark": (_build_drawdown_periods(b_dates, b_eq) if benchmark_data else None),
    }
    allocations_data = _build_allocations_data(out, benchmark)

    s_aligned, b_aligned = _aligned_return_series(out)
    strat_vals = _compute_metric_bundle(s_aligned, b_aligned)
    bench_vals = _compute_metric_bundle(b_aligned, None)
    if not strat_vals:
        raise ValueError("Unable to compute strategy metrics for report.")
    if not bench_vals:
        bench_vals = {}

    full_metrics = {
        "strategy": {
            "sharpe": strat_vals.get("sharpe"),
            "sortino": strat_vals.get("sortino"),
            "volatility": strat_vals.get("vol_ann"),
            "cagr": strat_vals.get("cagr"),
        },
        "benchmark": {
            "sharpe": bench_vals.get("sharpe"),
            "sortino": bench_vals.get("sortino"),
            "volatility": bench_vals.get("vol_ann"),
            "cagr": bench_vals.get("cagr"),
        } if benchmark_data else None,
    }

    rendered_right = _render_right_column(out.strategy_name, benchmark, metric_meta, strat_vals, bench_vals)

    raw = template
    raw = re.sub(r"<title>.*?</title>", f"<title>{html.escape(out.strategy_name)}</title>", raw, count=1, flags=re.S)
    raw = re.sub(r"<h1>Report for .*?</h1>", f"<h1>Report for {html.escape(out.strategy_name)}</h1>", raw, count=1, flags=re.S)
    subtitle = (
        f"Generated on {dt.datetime.now().strftime('%b %d, %Y at %H:%M:%S')} "
        f"for the period from <b>{_fmt_long_date(out.start_date)}</b> "
        f"to <b>{_fmt_long_date(out.end_date)}</b><br>Benchmark is <b>{html.escape(benchmark)}</b>"
    )
    raw = re.sub(r"<div class=\"subtitle\">.*?</div>", f"<div class=\"subtitle\">{subtitle}</div>", raw, count=1, flags=re.S)
    raw = re.sub(
        r"(<div id=\"strategy-name-desc\" class=\"sr-only\">).*?(</div>)",
        r"\1" + html.escape(out.strategy_name) + r"\2",
        raw,
        count=1,
        flags=re.S,
    )
    raw = re.sub(
        r"(<div id=\"benchmark-name-desc\" class=\"sr-only\">).*?(</div>)",
        r"\1" + html.escape(benchmark) + r"\2",
        raw,
        count=1,
        flags=re.S,
    )

    raw = re.sub(
        r"<div class=\"right-column\">.*?</div>\s*</div>\s*<script defer src=\"https://cdn\.plot\.ly/plotly-2\.35\.2\.min\.js\"></script>",
        rendered_right + "\n</div>\n<script defer src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>",
        raw,
        count=1,
        flags=re.S,
    )

    raw = _replace_js_const(raw, "strategyData", strategy_data)
    raw = _replace_js_const(raw, "benchmarkData", benchmark_data)
    raw = _insert_js_const_after(raw, "benchmarkData", "extraBenchmarkData", chart_benchmark_data)
    raw = _replace_js_const(raw, "rollingData", rolling_data)
    raw = _replace_js_const(raw, "monthlyHeatmapData", monthly_heatmap_data)
    raw = _replace_js_const(raw, "eoyData", eoy_data)
    raw = _replace_js_const(raw, "drawdownPeriods", drawdown_periods)
    raw = _replace_js_const(raw, "allocationsData", allocations_data)
    raw = _replace_js_const(raw, "fullMetrics", full_metrics)
    if chart_benchmark_data:
        raw = _patch_cumulative_chart_for_extra_benchmark(raw, out.chart_benchmark_name)

    path.write_text(raw, encoding="utf-8")


def _append_warnings_to_report(path: Path, out: BacktestOutput) -> None:
    if not out.warnings:
        return
    warnings_html = "".join(f"<li>{html.escape(w)}</li>" for w in out.warnings[:1000])
    section = (
        "<section style='margin:24px 0 8px 0;padding:16px;border:1px solid #ddd;border-radius:8px;'>"
        f"<h2 style='margin-top:0'>Engine Warnings ({len(out.warnings)})</h2>"
        f"<ul style='margin:0'>{warnings_html}</ul>"
        "</section>"
    )
    raw = path.read_text(encoding="utf-8")
    if "</body>" in raw:
        raw = raw.replace("</body>", section + "\n</body>")
    else:
        raw += "\n" + section
    path.write_text(raw, encoding="utf-8")


def _append_diagnostics_to_report(path: Path, out: BacktestOutput) -> None:
    limiting = ", ".join(out.limiting_symbols) if out.limiting_symbols else "(none)"
    src = out.data_source or {}
    net_calls = "yes" if src.get("network_calls") else "no"
    api_enabled = "yes" if src.get("testfolio_api_enabled") else "no"
    auth_supplied = "yes" if src.get("testfolio_auth_supplied") else "no"
    auth_required = "yes" if src.get("testfolio_auth_required") else "no"
    disabled_reason = str(src.get("testfolio_disabled_reason") or "")
    coverage_rows = []
    for r in sorted(
        out.symbol_coverage,
        key=lambda x: (x.get("first_date") is None, x.get("first_date") or "", x.get("symbol") or ""),
    ):
        sym = str(r.get("symbol") or "")
        first_date = r.get("first_date") or "-"
        last_date = r.get("last_date") or "-"
        proxied = r.get("proxied_to") or "-"
        missing = "yes" if r.get("missing") else "no"
        limits = "yes" if sym in set(out.limiting_symbols) else "no"
        coverage_rows.append(
            "<tr>"
            f"<td>{html.escape(sym)}</td>"
            f"<td>{html.escape(first_date)}</td>"
            f"<td>{html.escape(last_date)}</td>"
            f"<td>{html.escape(proxied)}</td>"
            f"<td>{html.escape(missing)}</td>"
            f"<td>{html.escape(limits)}</td>"
            "</tr>"
        )

    section = (
        "<section style='margin:24px 0;padding:16px;border:1px solid #ddd;border-radius:8px;'>"
        "<h2 style='margin-top:0'>Backtest Diagnostics</h2>"
        "<p style='margin:0 0 8px 0;'>"
        f"<b>Effective start:</b> {out.start_date.isoformat()} | "
        f"<b>Raw symbol start:</b> {out.raw_start_date.isoformat()} | "
        f"<b>Warmup:</b> {out.warmup_days} trading days | "
        f"<b>Limiting symbol(s):</b> {html.escape(limiting)}"
        "</p>"
        "<p style='margin:0 0 8px 0;'>"
        f"<b>Data source:</b> {html.escape(str(src.get('type') or 'unknown'))} "
        f"(config: {html.escape(str(src.get('config_dir') or ''))}; "
        f"tickers={html.escape(str(src.get('tickers_files') or 0))}, "
        f"ext_letf={html.escape(str(src.get('ext_letf_files') or 0))}, "
        f"prices={html.escape(str(src.get('prices_files') or 0))}; "
        f"testfolio API enabled: {api_enabled}; "
        f"testfolio auth supplied: {auth_supplied}; "
        f"testfolio auth required: {auth_required}; "
        f"network/API calls: {net_calls}; "
        f"api calls={html.escape(str(src.get('testfolio_api_calls') or 0))}; "
        f"api failures={html.escape(str(src.get('testfolio_api_failures') or 0))}; "
        f"yahoo fallback enabled: {'yes' if src.get('yahoo_fallback_enabled') else 'no'}; "
        f"yahoo calls={html.escape(str(src.get('yahoo_api_calls') or 0))}; "
        f"yahoo failures={html.escape(str(src.get('yahoo_api_failures') or 0))}"
        + (f"; testfolio disabled reason: {html.escape(disabled_reason)}" if disabled_reason else "")
        + ")"
        "</p>"
        "<details><summary>Per-symbol availability</summary>"
        "<div style='overflow:auto;max-height:380px;margin-top:8px;'>"
        "<table style='border-collapse:collapse;width:100%;font-size:12px;'>"
        "<thead><tr>"
        "<th style='border:1px solid #ddd;padding:4px;'>Symbol</th>"
        "<th style='border:1px solid #ddd;padding:4px;'>First Date</th>"
        "<th style='border:1px solid #ddd;padding:4px;'>Last Date</th>"
        "<th style='border:1px solid #ddd;padding:4px;'>Proxied To</th>"
        "<th style='border:1px solid #ddd;padding:4px;'>Missing</th>"
        "<th style='border:1px solid #ddd;padding:4px;'>Limits Start</th>"
        "</tr></thead><tbody>"
        + "".join(coverage_rows)
        + "</tbody></table></div></details></section>"
    )
    raw = path.read_text(encoding="utf-8")
    if "</body>" in raw:
        raw = raw.replace("</body>", section + "\n</body>")
    else:
        raw += "\n" + section
    path.write_text(raw, encoding="utf-8")


def _write_report_html_simple(path: Path, out: BacktestOutput, benchmark: str) -> None:
    m = out.metrics
    benchmark_lookup = dict(out.benchmark_curve)
    rows_eq = [
        [
            d.isoformat(),
            round(float(v), 10),
            (round(float(benchmark_lookup[d]), 10) if d in benchmark_lookup and math.isfinite(float(benchmark_lookup[d])) else None),
        ]
        for d, v in out.equity_curve
    ]
    warnings_html = "".join(f"<li>{html.escape(w)}</li>" for w in out.warnings[:500])
    def pct(key: str) -> str:
        return f"{m.get(key, 0.0) * 100:.2f}%"

    def num(key: str) -> str:
        return f"{m.get(key, 0.0):.2f}"

    raw = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>{html.escape(out.strategy_name)} - QM Native Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #172033; background: linear-gradient(180deg, #f6f8fc 0%, #eef3f8 100%); }}
.page {{ max-width: 1240px; margin: 0 auto; padding: 32px 24px 56px; }}
.hero {{ padding: 24px 28px; border: 1px solid #d9e2ef; border-radius: 18px; background: linear-gradient(135deg, #ffffff 0%, #eef5ff 100%); box-shadow: 0 10px 30px rgba(24, 44, 84, 0.08); }}
h1 {{ margin: 0 0 8px; font-size: 32px; line-height: 1.1; }}
.subtitle {{ color: #52627a; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 14px; margin: 18px 0 24px; }}
.card {{ border: 1px solid #d9e2ef; border-radius: 14px; padding: 14px 16px; background: rgba(255,255,255,0.88); box-shadow: 0 6px 18px rgba(24, 44, 84, 0.05); }}
.label {{ font-size: 12px; color: #607089; text-transform: uppercase; letter-spacing: 0.04em; }}
.value {{ font-size: 24px; font-weight: 700; margin-top: 6px; }}
.panel {{ margin-top: 20px; border: 1px solid #d9e2ef; border-radius: 18px; padding: 18px; background: #fff; box-shadow: 0 8px 22px rgba(24, 44, 84, 0.05); }}
.panel h2, .panel h3 {{ margin: 0 0 12px; }}
#chart {{ width: 100%; height: 520px; display: block; }}
.legend {{ display: flex; gap: 18px; margin: 10px 0 0; color: #52627a; font-size: 13px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
.swatch {{ width: 14px; height: 3px; border-radius: 999px; }}
ul {{ margin: 0; padding-left: 20px; }}
small {{ color: #666; }}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }}
}}
@media (max-width: 560px) {{
  .page {{ padding: 18px 14px 36px; }}
  .hero {{ padding: 18px; }}
  .grid {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 26px; }}
}}
</style>
</head>
<body>
<div class=\"page\">
  <section class=\"hero\">
    <h1>{html.escape(out.strategy_name)}</h1>
    <div class=\"subtitle\">Backtest range: {out.start_date.isoformat()} to {out.end_date.isoformat()} | Benchmark: {html.escape(benchmark)}</div>
    <div class=\"grid\">
      <div class=\"card\"><div class=\"label\">Total Return</div><div class=\"value\">{pct('total_return')}</div></div>
      <div class=\"card\"><div class=\"label\">CAGR</div><div class=\"value\">{pct('cagr')}</div></div>
      <div class=\"card\"><div class=\"label\">Max Drawdown</div><div class=\"value\">{pct('max_drawdown')}</div></div>
      <div class=\"card\"><div class=\"label\">Volatility</div><div class=\"value\">{pct('volatility')}</div></div>
      <div class=\"card\"><div class=\"label\">Sharpe</div><div class=\"value\">{num('sharpe')}</div></div>
    </div>
  </section>
  <section class=\"panel\">
    <h2>Equity Curve</h2>
    <canvas id=\"chart\"></canvas>
    <div class=\"legend\">
      <span><i class=\"swatch\" style=\"background:#1f6feb\"></i>Strategy</span>
      <span><i class=\"swatch\" style=\"background:#f97316\"></i>Benchmark</span>
    </div>
  </section>
  <section class=\"panel\">
    <h3>Warnings ({len(out.warnings)})</h3>
    <ul>{warnings_html}</ul>
  </section>
</div>
<script>
const data = {json.dumps(rows_eq, separators=(',', ':'))};
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
function draw() {{
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * window.devicePixelRatio;
  canvas.height = h * window.devicePixelRatio;
  ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const vals = data.map(r => [r[1], r[2]]).flat().filter(v => Number.isFinite(v));
  const minV = Math.min(...vals), maxV = Math.max(...vals);
  const padL = 64, padR = 20, padT = 18, padB = 42;
  const cw = w - padL - padR, ch = h - padT - padB;
  const x = i => padL + (i / (data.length - 1)) * cw;
  const y = v => (maxV === minV) ? padT + ch / 2 : padT + (1 - (v - minV) / (maxV - minV)) * ch;
  ctx.strokeStyle = '#d8e0ea'; ctx.lineWidth = 1; ctx.beginPath();
  ctx.moveTo(padL, padT + ch); ctx.lineTo(padL + cw, padT + ch);
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + ch); ctx.stroke();
  ctx.fillStyle = '#607089';
  ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  for (let i = 0; i <= 4; i++) {{
    const val = minV + ((maxV - minV) * (4 - i) / 4);
    const yy = padT + (ch * i / 4);
    ctx.strokeStyle = '#eef2f7';
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(padL + cw, yy);
    ctx.stroke();
    ctx.fillText(val.toFixed(2), 8, yy + 4);
  }}
  ctx.strokeStyle = '#1f6feb'; ctx.lineWidth = 2.5; ctx.beginPath();
  data.forEach((r, i) => {{ const xx = x(i), yy = y(r[1]); if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); }});
  ctx.stroke();
  ctx.strokeStyle = '#f97316'; ctx.lineWidth = 2.5; ctx.beginPath();
  let benchStarted = false;
  data.forEach((r, i) => {{
    const xx = x(i), yy = y(r[2]);
    if (!Number.isFinite(r[2])) return;
    if (!benchStarted) {{
      ctx.moveTo(xx, yy);
      benchStarted = true;
    }} else {{
      ctx.lineTo(xx, yy);
    }}
  }});
  ctx.stroke();
}}
window.addEventListener('resize', draw); draw();
</script>
</body>
</html>
"""
    path.write_text(raw, encoding="utf-8")


def write_report_html(path: Path, out: BacktestOutput, benchmark: str) -> None:
    try:
        mpl_dir = (Path(tempfile.gettempdir()) / "mpl_cache_qm_native").resolve()
        xdg_dir = (Path(tempfile.gettempdir()) / "xdg_cache_qm_native").resolve()
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
        mpl_dir.mkdir(parents=True, exist_ok=True)
        xdg_dir.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _write_report_html_rain_style(path, out, benchmark)
        _append_diagnostics_to_report(path, out)
        _append_warnings_to_report(path, out)
    except Exception as exc:
        out.warnings.append(f"Rain-style report generation failed; using simple HTML fallback. Reason: {exc}")
        _write_report_html_simple(path, out, benchmark)
        _append_diagnostics_to_report(path, out)
        _append_warnings_to_report(path, out)


# ----------------------------- Strategy discovery -----------------------------


def looks_like_strategy(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if "incantation" not in node:
        return False
    hints = {"name", "trading_type", "threshold", "benchmark_ticker", "version"}
    return any(k in node for k in hints)


def collect_strategies(node: Any) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []

    def walk(cur: Any, path: str) -> None:
        if isinstance(cur, dict):
            if looks_like_strategy(cur):
                out.append((path, cur))
            for k, v in cur.items():
                walk(v, f"{path}.{k}")
        elif isinstance(cur, list):
            for i, v in enumerate(cur):
                walk(v, f"{path}[{i}]")

    walk(node, "$")
    return out


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return s or "strategy"


# ----------------------------- CLI -----------------------------


def run_cli() -> int:
    p = argparse.ArgumentParser(description="Quantmage-native backtest on Testfolio cache + API extension")
    p.add_argument("-i", "--input", required=True, help="Path to Quantmage JSON (single strategy or export)")
    p.add_argument("-o", "--output", default="reports/qm_native", help="Output file (single strategy) or directory (multi)")
    p.add_argument("--config", default="config", help="Config directory with MarketDays.dat and price caches")
    p.add_argument("--benchmark", default=None, help="Override benchmark ticker")
    p.add_argument("--chart-benchmark", default="", help="Optional extra ticker shown only in the cumulative returns chart")
    p.add_argument("--strict", action="store_true", help="Fail when compiler emits warnings")
    p.add_argument("--no-testfolio-api", action="store_true", help="Disable Testfolio API extension (cache-only mode)")
    p.add_argument("--no-yahoo-fallback", action="store_true", help="Disable Yahoo Finance fallback fetches")
    p.add_argument("--allow-non-letf-proxy", action="store_true", help="Allow heuristic/non-LetfMap proxy substitutions")
    p.add_argument("--testfolio-sim-jar", default="", help="Optional BacktestReport.jar path for local Testfolio_SIM.csv history")
    p.add_argument("--refresh-sim-cache", action="store_true", help="Rebuild config/extended_prices/testfolio-sim.csv from BacktestReport.jar before running")
    p.add_argument("--extended-price-csv", action="append", default=[], help="Optional wide CSV with Date column and one symbol per column")
    p.add_argument("--testfolio-api-url", default="https://testfol.io/api/tactical", help="Testfolio tactical endpoint")
    p.add_argument("--testfolio-timeout", type=float, default=40.0, help="Testfolio API timeout in seconds")
    p.add_argument("--testfolio-retries", type=int, default=2, help="Retries for Testfolio API requests")
    p.add_argument("--testfolio-cookie", default="", help="Optional Cookie header value for authenticated Testfolio access")
    p.add_argument("--testfolio-token", default="", help="Optional bearer token for authenticated Testfolio access")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    cfg = Path(args.config).expanduser().resolve()

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    strategies = collect_strategies(payload)
    if not strategies:
        print("No Quantmage strategy objects detected in input.")
        return 2

    store = DataStore(
        cfg,
        use_testfolio_api=not args.no_testfolio_api,
        use_yahoo_fallback=not args.no_yahoo_fallback,
        allow_non_letf_proxy=bool(args.allow_non_letf_proxy),
        testfolio_sim_jar=Path(args.testfolio_sim_jar).expanduser() if args.testfolio_sim_jar else None,
        refresh_sim_cache=bool(args.refresh_sim_cache),
        extended_price_csvs=tuple(Path(p).expanduser() for p in (args.extended_price_csv or [])),
        testfolio_cookie=str(args.testfolio_cookie or os.environ.get("TESTFOLIO_COOKIE", "")),
        testfolio_token=str(args.testfolio_token or os.environ.get("TESTFOLIO_TOKEN", "")),
        testfolio_api_url=str(args.testfolio_api_url),
        testfolio_timeout_sec=max(1.0, float(args.testfolio_timeout)),
        testfolio_retries=max(0, int(args.testfolio_retries)),
    )
    bt = QMBacktester(store, strict=args.strict)

    if len(strategies) == 1:
        _path, qm = strategies[0]
        out_dir = out_path.parent if out_path.suffix else out_path
        ensure_dir(out_dir)
        name = str(qm.get("name") or "strategy")
        slug = slugify(name)

        result = bt.run(qm, benchmark=args.benchmark, chart_benchmark=args.chart_benchmark)
        alloc_csv = out_dir / f"{slug}.allocation.csv"
        equity_csv = out_dir / f"{slug}.equity.csv"
        symbols_csv = out_dir / f"{slug}.symbols.csv"
        limiting_csv = out_dir / f"{slug}.limiting_symbols.csv"
        start_txt = out_dir / f"{slug}.start_diagnostics.txt"
        report_html = out_dir / f"{slug}.report.html"
        warn_txt = out_dir / f"{slug}.warnings.txt"

        write_allocation_csv(alloc_csv, result.allocations)
        write_equity_csv(equity_csv, result.equity_curve, result.benchmark_curve)
        write_symbol_coverage_csv(symbols_csv, result)
        write_limiting_symbols_csv(limiting_csv, result)
        write_start_diagnostics_txt(start_txt, result)
        write_report_html(report_html, result, args.benchmark or str(qm.get("benchmark_ticker") or "SPY").upper())
        if result.warnings:
            warn_txt.write_text("\n".join(result.warnings) + "\n", encoding="utf-8")
        elif warn_txt.exists():
            warn_txt.unlink()

        print(f"Strategy: {result.strategy_name}")
        print(f"Backtest: {result.start_date} -> {result.end_date}")
        print(f"CAGR: {result.metrics.get('cagr', 0.0) * 100:.2f}%")
        print(f"MaxDD: {result.metrics.get('max_drawdown', 0.0) * 100:.2f}%")
        print(f"Sharpe: {result.metrics.get('sharpe', 0.0):.2f}")
        print(f"Allocation CSV: {alloc_csv}")
        print(f"Equity CSV: {equity_csv}")
        print(f"Symbols CSV: {symbols_csv}")
        print(f"Limiting Symbols CSV: {limiting_csv}")
        print(f"Start Diagnostics: {start_txt}")
        print(f"Report HTML: {report_html}")
        if result.warnings:
            print(f"Warnings: {warn_txt} ({len(result.warnings)})")
        return 0

    # Multi strategy
    if out_path.suffix:
        print("For multi-strategy input, --output must be a directory path.")
        return 2
    ensure_dir(out_path)

    index_items: List[Dict[str, Any]] = []
    for i, (_path, qm) in enumerate(strategies, start=1):
        name = str(qm.get("name") or f"strategy-{i}")
        slug = slugify(name)
        # disambiguate duplicates
        while (out_path / f"{slug}.report.html").exists():
            slug = f"{slug}-{i}"

        result = bt.run(qm, benchmark=args.benchmark, chart_benchmark=args.chart_benchmark)
        alloc_csv = out_path / f"{slug}.allocation.csv"
        equity_csv = out_path / f"{slug}.equity.csv"
        symbols_csv = out_path / f"{slug}.symbols.csv"
        limiting_csv = out_path / f"{slug}.limiting_symbols.csv"
        start_txt = out_path / f"{slug}.start_diagnostics.txt"
        report_html = out_path / f"{slug}.report.html"
        warn_txt = out_path / f"{slug}.warnings.txt"

        write_allocation_csv(alloc_csv, result.allocations)
        write_equity_csv(equity_csv, result.equity_curve, result.benchmark_curve)
        write_symbol_coverage_csv(symbols_csv, result)
        write_limiting_symbols_csv(limiting_csv, result)
        write_start_diagnostics_txt(start_txt, result)
        write_report_html(report_html, result, args.benchmark or str(qm.get("benchmark_ticker") or "SPY").upper())
        if result.warnings:
            warn_txt.write_text("\n".join(result.warnings) + "\n", encoding="utf-8")
        elif warn_txt.exists():
            warn_txt.unlink()

        index_items.append(
            {
                "name": result.strategy_name,
                "start": result.start_date.isoformat(),
                "end": result.end_date.isoformat(),
                "cagr": result.metrics.get("cagr", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "sharpe": result.metrics.get("sharpe", 0.0),
                "allocation_csv": alloc_csv.name,
                "equity_csv": equity_csv.name,
                "symbols_csv": symbols_csv.name,
                "limiting_symbols_csv": limiting_csv.name,
                "start_diagnostics_txt": start_txt.name,
                "report_html": report_html.name,
                "warnings_txt": warn_txt.name if result.warnings else None,
                "warning_count": len(result.warnings),
            }
        )

    idx_path = out_path / "index.json"
    idx_path.write_text(json.dumps({"count": len(index_items), "items": index_items}, indent=2), encoding="utf-8")
    print(f"Wrote {len(index_items)} strategy reports to {out_path}")
    print(f"Index: {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
