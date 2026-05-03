#!/usr/bin/env python3
"""
Convert a Quantmage strategy JSON into Rain's Symphony JSON format.

This adapter does not modify BacktestReport.jar. It generates a JSON file that
the existing tool can already load (Strategy Input -> from file / direct input).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


INDICATOR_MAP = {
    "CumulativeReturn": "cumulative-return",
    "RelativeStrengthIndex": "relative-strength-index",
    "Volatility": "standard-deviation-return",
    "13612wMomentum": "moving-average-return",
    "13612uMomentum": "moving-average-return",
    "Sma12Momentum": "moving-average-return",
    "MovingAverageOfReturns": "moving-average-return",
    "MovingAverage": "moving-average-price",
    "ExponentialMovingAverage": "exponential-moving-average-price",
    "MaxDrawdown": "max-drawdown",
    "Drawdown": "max-drawdown",
    "CurrentPrice": "current-price",
}

REBALANCE_MAP = {
    "Daily": "DAILY",
    "Weekly": "WEEKLY",
    "Monthly": "MONTHLY",
    "Quarterly": "QUARTERLY",
    "Yearly": "YEARLY",
    "Threshold": "THRESHOLD",
}

# Quantmage macro series are not tradable tickers in Rain/Testfolio.
# Map known macros to tradable proxy ETFs; unknown macros fall back safely.
QM_MACRO_MAP = {
    # FRED high-yield spread proxy
    "@BAMLH0A0HYM2": "HYG",
    # FRED curve spread proxies
    "@T10Y3M": "TLT",
    "@T10Y2Y": "TLT",
    # FRED volatility series proxies
    "@VIXCLS": "UVXY",
    "@VXVCLS": "VIXM",
}

VALID_TICKER = re.compile(r"^[A-Z@][A-Z0-9._-]{0,19}$")


@dataclass
class ConvertContext:
    fallback_symbol: str = "BIL"
    warnings: List[str] = field(default_factory=list)

    def warn(self, path: str, message: str) -> None:
        self.warnings.append(f"{path}: {message}")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "quantmage-strategy"


def is_number_string(value: str) -> bool:
    return bool(re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", value))


def sanitize_window(window: Any, expr_fn: str) -> int:
    if expr_fn == "current-price":
        if window is None:
            return 0
        try:
            return max(0, int(window))
        except Exception:
            return 0
    try:
        w = int(window)
    except Exception:
        w = 0
    return max(1, w)


def fraction_from_weight(value: Any) -> Dict[str, int]:
    try:
        frac = Fraction(str(value)).limit_denominator(1_000_000)
    except Exception:
        frac = Fraction(0, 1)
    return {"num": int(frac.numerator), "den": int(frac.denominator)}


def normalize_ticker(raw: Any, ctx: ConvertContext, path: str) -> str:
    if raw is None:
        ctx.warn(path, f"missing ticker, using fallback '{ctx.fallback_symbol}'")
        return ctx.fallback_symbol
    if not isinstance(raw, str):
        ctx.warn(path, f"non-string ticker '{raw}', using fallback '{ctx.fallback_symbol}'")
        return ctx.fallback_symbol
    ticker = raw.strip()
    if not ticker:
        ctx.warn(path, f"empty ticker, using fallback '{ctx.fallback_symbol}'")
        return ctx.fallback_symbol
    if is_number_string(ticker):
        return ticker
    if ticker.startswith("Subspell") or "/" in ticker:
        ctx.warn(path, f"unsupported pseudo ticker '{ticker}', using fallback '{ctx.fallback_symbol}'")
        return ctx.fallback_symbol
    ticker = ticker.upper()
    if ticker.startswith("@"):
        mapped = QM_MACRO_MAP.get(ticker)
        if mapped:
            if VALID_TICKER.fullmatch(mapped):
                ctx.warn(path, f"macro ticker '{ticker}' mapped to proxy '{mapped}'")
                return mapped
            ctx.warn(path, f"macro ticker '{ticker}' map target '{mapped}' is invalid; using fallback '{ctx.fallback_symbol}'")
            return ctx.fallback_symbol
        ctx.warn(path, f"unsupported macro ticker '{ticker}', using fallback '{ctx.fallback_symbol}'")
        return ctx.fallback_symbol
    if not VALID_TICKER.fullmatch(ticker):
        ctx.warn(path, f"invalid ticker '{ticker}', using fallback '{ctx.fallback_symbol}'")
        return ctx.fallback_symbol
    return ticker


def map_indicator(indicator: Dict[str, Any], ctx: ConvertContext, path: str) -> str:
    ind_type = indicator.get("type")
    mapped = INDICATOR_MAP.get(ind_type)
    if mapped is None:
        ctx.warn(path, f"unsupported indicator '{ind_type}', mapped to 'cumulative-return'")
        return "cumulative-return"
    return mapped


def convert_single_condition(cond: Dict[str, Any], ctx: ConvertContext, path: str) -> Dict[str, Any]:
    cond_kind = cond.get("type", "IndicatorAndNumber")
    greater_than = bool(cond.get("greater_than", True))
    comparator = "GT" if greater_than else "LT"

    lhs_ind = cond.get("lh_indicator") or {}
    lhs_fn = map_indicator(lhs_ind, ctx, f"{path}.lh_indicator")
    lhs_window = sanitize_window(lhs_ind.get("window"), lhs_fn)
    lhs_val = normalize_ticker(cond.get("lh_ticker_symbol"), ctx, f"{path}.lh_ticker_symbol")

    if cond_kind == "BothIndicators":
        rhs_ind = cond.get("rh_indicator") or {}
        rhs_fn = map_indicator(rhs_ind, ctx, f"{path}.rh_indicator")
        rhs_window = sanitize_window(rhs_ind.get("window"), rhs_fn)
        rhs_val = normalize_ticker(cond.get("rh_ticker_symbol"), ctx, f"{path}.rh_ticker_symbol")
        if float(cond.get("rh_bias", 0.0) or 0.0) != 0.0:
            ctx.warn(path, "rh_bias is not supported and was ignored")
    else:
        if cond_kind in {"Month", "IndicatorPercentile", "IndicatorStochastic"}:
            ctx.warn(path, f"condition type '{cond_kind}' approximated as indicator-vs-number")
        rhs_fn = None
        rhs_window = None
        if cond_kind == "Month":
            rhs_val = int(cond.get("rh_value_int", 1) or 1)
        else:
            rhs_val = float(cond.get("rh_value", 0.0) or 0.0)

    for_days = int(cond.get("for_days", 1) or 1)
    if for_days != 1:
        ctx.warn(path, f"for_days={for_days} not supported and was ignored")
    if int(cond.get("lh_days_ago", 0) or 0) != 0 or int(cond.get("rh_days_ago", 0) or 0) != 0:
        ctx.warn(path, "days_ago offsets are not supported and were ignored")

    out: Dict[str, Any] = {
        "comparator": comparator,
        "lhs-fn": lhs_fn,
        "lhs-val": lhs_val,
    }
    if lhs_fn != "current-price":
        out["lhs-window-days"] = lhs_window

    if rhs_fn is None:
        out["rhs-val"] = rhs_val
    else:
        out["rhs-fn"] = rhs_fn
        out["rhs-val"] = rhs_val
        if rhs_fn != "current-price":
            out["rhs-window-days"] = rhs_window
    return out


def expand_boolean_condition(
    condition: Dict[str, Any],
    then_inc: Any,
    else_inc: Any,
    ctx: ConvertContext,
    path: str,
) -> Dict[str, Any]:
    cond_type = condition.get("condition_type")
    conds = condition.get("conditions") or []
    if not conds:
        ctx.warn(path, f"{cond_type or 'Unknown'} with empty conditions; using else branch")
        return {"incantation_type": "IfElse", "condition": {"condition_type": "SingleCondition", "type": "IndicatorAndNumber", "greater_than": True, "lh_indicator": {"type": "CurrentPrice", "window": 0}, "lh_ticker_symbol": ctx.fallback_symbol, "rh_value": -1.0}, "then_incantation": then_inc, "else_incantation": else_inc}

    if cond_type == "AllOf":
        nested = then_inc
        for idx in range(len(conds) - 1, 0, -1):
            nested = {
                "incantation_type": "IfElse",
                "condition": conds[idx],
                "then_incantation": nested,
                "else_incantation": else_inc,
            }
        return {
            "incantation_type": "IfElse",
            "condition": conds[0],
            "then_incantation": nested,
            "else_incantation": else_inc,
        }

    # AnyOf
    nested = else_inc
    for idx in range(len(conds) - 1, 0, -1):
        nested = {
            "incantation_type": "IfElse",
            "condition": conds[idx],
            "then_incantation": then_inc,
            "else_incantation": nested,
        }
    return {
        "incantation_type": "IfElse",
        "condition": conds[0],
        "then_incantation": then_inc,
        "else_incantation": nested,
    }


def convert_incantation(inc: Any, ctx: ConvertContext, path: str) -> Dict[str, Any]:
    if inc is None:
        ctx.warn(path, f"null branch replaced with fallback '{ctx.fallback_symbol}'")
        return {"step": "asset", "ticker": ctx.fallback_symbol}
    if not isinstance(inc, dict):
        ctx.warn(path, f"invalid incantation '{inc}', replaced with fallback '{ctx.fallback_symbol}'")
        return {"step": "asset", "ticker": ctx.fallback_symbol}

    inc_type = inc.get("incantation_type")
    if inc_type == "Ticker":
        ticker = normalize_ticker(inc.get("symbol"), ctx, f"{path}.symbol")
        name = (inc.get("name") or "").strip()
        node: Dict[str, Any] = {"step": "asset", "ticker": ticker}
        if name:
            node["name"] = name
        return node

    if inc_type == "Weighted":
        raw_children = inc.get("incantations") or []
        children = [
            convert_incantation(child, ctx, f"{path}.incantations[{idx}]")
            for idx, child in enumerate(raw_children)
            if child is not None
        ]
        if not children:
            ctx.warn(path, "weighted block has no children; using fallback asset")
            children = [{"step": "asset", "ticker": ctx.fallback_symbol}]

        qtype = (inc.get("type") or "Equal").strip()
        if qtype == "InverseVolatility":
            step = "wt-inverse-vol"
        elif qtype == "Custom":
            step = "wt-cash-specified"
        else:
            step = "wt-cash-equal"

        entries: List[Dict[str, Any]] = []
        weights = inc.get("weights") or []
        for idx, child in enumerate(children):
            entry = deepcopy(child)
            if step == "wt-cash-specified":
                w = weights[idx] if idx < len(weights) else 0
                entry["weight"] = fraction_from_weight(w)
            entries.append(entry)

        node = {"step": step, "children": entries}
        if step == "wt-inverse-vol":
            node["window-days"] = max(1, int(inc.get("inverse_volatility_window") or 1))
        return node

    if inc_type == "Filtered":
        raw_children = inc.get("incantations") or []
        children = [
            convert_incantation(child, ctx, f"{path}.incantations[{idx}]")
            for idx, child in enumerate(raw_children)
            if child is not None
        ]
        if not children:
            ctx.warn(path, "filter block has no children; using fallback asset")
            children = [{"step": "asset", "ticker": ctx.fallback_symbol}]

        sort_indicator = inc.get("sort_indicator") or {}
        sort_fn = map_indicator(sort_indicator, ctx, f"{path}.sort_indicator")
        sort_window = sanitize_window(sort_indicator.get("window"), sort_fn)
        select_fn = "BOTTOM" if bool(inc.get("bottom", False)) else "TOP"
        select_n = max(1, int(inc.get("count") or 1))

        weight_type = (inc.get("weight_type") or "Equal").strip()
        if weight_type != "Equal":
            ctx.warn(path, f"filter weight_type '{weight_type}' is not supported and was ignored")
        if (inc.get("inverse_volatility_window") or 0) not in (0, None):
            ctx.warn(path, "filter inverse volatility weighting is not supported and was ignored")

        return {
            "step": "filter",
            "sort-by-fn": sort_fn,
            "sort-by-window-days": sort_window,
            "select-fn": select_fn,
            "select-n": select_n,
            "children": children,
        }

    if inc_type == "IfElse":
        condition = inc.get("condition") or {}
        cond_type = condition.get("condition_type")
        then_inc = inc.get("then_incantation")
        else_inc = inc.get("else_incantation")

        if cond_type in {"AllOf", "AnyOf"}:
            expanded = expand_boolean_condition(condition, then_inc, else_inc, ctx, path)
            return convert_incantation(expanded, ctx, f"{path}.expanded")

        if cond_type != "SingleCondition":
            ctx.warn(path, f"unsupported condition_type '{cond_type}', using else branch")
            return convert_incantation(else_inc, ctx, f"{path}.else_incantation")

        condition_map = convert_single_condition(condition, ctx, f"{path}.condition")
        then_node = convert_incantation(then_inc, ctx, f"{path}.then_incantation")
        else_node = convert_incantation(else_inc, ctx, f"{path}.else_incantation")

        return {
            "step": "if",
            "children": [
                {
                    "is-else-condition?": False,
                    **condition_map,
                    "children": [then_node],
                },
                {
                    "is-else-condition?": True,
                    "children": [else_node],
                },
            ],
        }

    if inc_type == "Switch":
        conds = inc.get("conditions") or []
        branches = inc.get("incantations") or []
        if not branches:
            ctx.warn(path, "switch has no branches; using fallback asset")
            return {"step": "asset", "ticker": ctx.fallback_symbol}
        default_branch = branches[len(conds)] if len(branches) > len(conds) else branches[-1]
        nested: Any = default_branch
        for idx in range(len(conds) - 1, -1, -1):
            then_branch = branches[idx] if idx < len(branches) else None
            nested = {
                "incantation_type": "IfElse",
                "condition": conds[idx],
                "then_incantation": then_branch,
                "else_incantation": nested,
            }
        ctx.warn(path, "switch converted to nested if/else chain")
        return convert_incantation(nested, ctx, f"{path}.expanded")

    if inc_type == "EnterExit":
        ctx.warn(path, "enter/exit stateful logic approximated as simple if/else")
        approx = {
            "incantation_type": "IfElse",
            "condition": inc.get("enter_condition"),
            "then_incantation": inc.get("enter_incantation"),
            "else_incantation": inc.get("exit_incantation"),
        }
        return convert_incantation(approx, ctx, f"{path}.expanded")

    if inc_type == "Mixed":
        ctx.warn(path, "mixed node is not supported; using 'to_incantation' branch")
        return convert_incantation(inc.get("to_incantation"), ctx, f"{path}.to_incantation")

    ctx.warn(path, f"unsupported incantation_type '{inc_type}', replaced with fallback asset")
    return {"step": "asset", "ticker": ctx.fallback_symbol}


def convert_strategy(qm: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = (qm.get("benchmark_ticker") or "BIL").strip().upper() or "BIL"
    fallback = benchmark if VALID_TICKER.fullmatch(benchmark) else "BIL"
    ctx = ConvertContext(fallback_symbol=fallback)

    root_inc = qm.get("incantation")
    root_node = convert_incantation(root_inc, ctx, "incantation")

    name = (qm.get("name") or "Quantmage Strategy").strip() or "Quantmage Strategy"
    trading_type = (qm.get("trading_type") or "Threshold").strip()
    rebalance = REBALANCE_MAP.get(trading_type, "THRESHOLD")
    if trading_type not in REBALANCE_MAP:
        ctx.warn("trading_type", f"unknown trading_type '{trading_type}', using 'THRESHOLD'")

    threshold = qm.get("threshold", 0)
    try:
        corridor = float(threshold)
    except Exception:
        corridor = 0.0

    out = {
        "step": "root",
        "name": name,
        "description": qm.get("description") or "",
        "id": slugify(name),
        "rebalance": rebalance,
        "rebalance-corridor-width": corridor,
        "children": [root_node],
    }

    return {"strategy": out, "warnings": ctx.warnings}


def looks_like_strategy(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if "incantation" not in node:
        return False
    inc = node.get("incantation")
    if inc is not None and not isinstance(inc, dict):
        return False
    strategy_hints = {"name", "trading_type", "benchmark_ticker", "threshold", "version", "slippage_bps"}
    return any(key in node for key in strategy_hints)


def collect_strategies(node: Any) -> List[Tuple[str, Dict[str, Any]]]:
    found: List[Tuple[str, Dict[str, Any]]] = []
    seen: Set[int] = set()

    def walk(curr: Any, path: str) -> None:
        ident = id(curr)
        if ident in seen:
            return
        seen.add(ident)
        if isinstance(curr, dict):
            if looks_like_strategy(curr):
                found.append((path, curr))
            for key, value in curr.items():
                next_path = f"{path}.{key}" if path else str(key)
                walk(value, next_path)
        elif isinstance(curr, list):
            for idx, value in enumerate(curr):
                next_path = f"{path}[{idx}]" if path else f"[{idx}]"
                walk(value, next_path)

    walk(node, "$")
    return found


def make_unique_slug(base_name: str, used: Set[str]) -> str:
    base = slugify(base_name)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def strict_blockers(warnings: List[str]) -> List[str]:
    blockers: List[str] = []
    for line in warnings:
        l = line.lower()
        if "unsupported " in l or " approximated " in f" {l} " or " not supported " in f" {l} ":
            blockers.append(line)
            continue
        if "replaced with fallback" in l or "mapped to 'cumulative-return'" in l:
            blockers.append(line)
    return blockers


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Optional[Path], payload: Any) -> None:
    content = json.dumps(payload, indent=2, ensure_ascii=True)
    if path is None:
        sys.stdout.write(content + "\n")
    else:
        path.write_text(content + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Quantmage JSON to Rain Symphony JSON")
    parser.add_argument("-i", "--input", required=True, help="Path to Quantmage strategy JSON")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for converted Symphony JSON (default: <input>.rain.json)",
    )
    parser.add_argument(
        "--warnings",
        help="Optional path to write conversion warnings (default: <output>.warnings.txt)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if unsupported/approximated logic is encountered during conversion.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 2

    qm_json = load_json(input_path)
    strategies = collect_strategies(qm_json)
    if not strategies:
        print("Input JSON does not contain any detectable Quantmage strategy objects.", file=sys.stderr)
        return 2

    converted: List[Tuple[str, Dict[str, Any], List[str]]] = []
    for idx, (path_hint, strategy_obj) in enumerate(strategies, start=1):
        result = convert_strategy(strategy_obj)
        name = (strategy_obj.get("name") or "").strip() or f"quantmage-strategy-{idx}"
        converted.append((name, result["strategy"], result["warnings"]))

        if args.strict:
            blockers = strict_blockers(result["warnings"])
            if blockers:
                print(f"Strict mode failed for strategy '{name}' at {path_hint}.", file=sys.stderr)
                for line in blockers[:20]:
                    print(f"  - {line}", file=sys.stderr)
                if len(blockers) > 20:
                    print(f"  ... and {len(blockers) - 20} more", file=sys.stderr)
                return 3

    # Single strategy keeps original behavior (one output file).
    if len(converted) == 1:
        name, strategy, warnings = converted[0]
        output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".rain.json")
        dump_json(output_path, strategy)

        warnings_path: Optional[Path]
        if args.warnings:
            warnings_path = Path(args.warnings).expanduser().resolve()
        else:
            warnings_path = output_path.with_suffix(output_path.suffix + ".warnings.txt")

        if warnings:
            warnings_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
            print(f"Wrote converted strategy: {output_path}")
            print(f"Wrote {len(warnings)} warnings: {warnings_path}")
        else:
            print(f"Wrote converted strategy: {output_path}")
            print("No conversion warnings.")
        return 0

    # Multi-strategy input writes one Rain JSON per strategy in an output directory.
    output_target = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".rain")
    if args.output and output_target.suffix:
        print(
            f"Input has {len(converted)} strategies. For multi-strategy conversion, --output must be a directory path (got file-like path: {output_target}).",
            file=sys.stderr,
        )
        return 2
    output_target.mkdir(parents=True, exist_ok=True)

    used_slugs: Set[str] = set()
    index_items: List[Dict[str, Any]] = []
    total_warnings = 0

    for name, strategy, warnings in converted:
        slug = make_unique_slug(name, used_slugs)
        strategy_path = output_target / f"{slug}.rain.json"
        warnings_path = output_target / f"{slug}.warnings.txt"

        dump_json(strategy_path, strategy)
        if warnings:
            warnings_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        total_warnings += len(warnings)

        index_items.append(
            {
                "name": name,
                "strategy_file": strategy_path.name,
                "warnings_file": warnings_path.name if warnings else None,
                "warning_count": len(warnings),
            }
        )

    index_path = output_target / "index.json"
    dump_json(
        index_path,
        {
            "input_file": str(input_path),
            "strategy_count": len(converted),
            "total_warnings": total_warnings,
            "items": index_items,
        },
    )

    print(f"Wrote {len(converted)} converted strategies to: {output_target}")
    print(f"Wrote summary index: {index_path}")
    print(f"Total warnings: {total_warnings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
