from __future__ import annotations

from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

from . import portfolio_lab as pl

TRADING_DAYS = 252


def _clean(r: pd.DataFrame) -> pd.DataFrame:
    return r.copy().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 5:
        return np.nan
    return float(x.iloc[:, 0].corr(x.iloc[:, 1]))


def _upper_mean(mat: pd.DataFrame) -> float:
    if mat.shape[0] < 2:
        return np.nan
    vals = mat.where(np.triu(np.ones(mat.shape), 1).astype(bool)).stack()
    return float(vals.mean()) if len(vals) else np.nan


def marginal_correlation_requirements(returns: pd.DataFrame, weights: pd.Series | Dict[str, float]) -> pd.DataFrame:
    """Estimate correlation thresholds that make each strategy worth adding.

    The thresholds are heuristic but actionable:
    - Sharpe accretion threshold: strategy improves portfolio Sharpe if rho < SR_i / SR_p.
    - Volatility reduction threshold: a tiny allocation can reduce portfolio variance if rho < sigma_p / sigma_i.
    Tail correlation is measured during portfolio drawdowns below the 10th percentile of drawdown.
    """
    r = _clean(returns).fillna(0.0)
    if r.empty:
        return pd.DataFrame()
    w = pl.normalize_weights(weights, list(r.columns))
    port = pl.portfolio_returns(r, w, "Active")
    p_m = pl.metrics_for_series(port)
    sr_p = float(p_m.get("Sharpe", np.nan))
    vol_p = float(p_m.get("Volatility", np.nan))
    dd = pl.drawdowns(port)
    trigger = dd <= dd.quantile(0.10) if len(dd) else pd.Series(False, index=r.index)

    rows = []
    for c in r.columns:
        s = r[c].dropna()
        m = pl.metrics_for_series(s)
        sr_i = float(m.get("Sharpe", np.nan))
        vol_i = float(m.get("Volatility", np.nan))
        corr_p = _safe_corr(r[c], port)
        tail_corr = _safe_corr(r.loc[trigger.index[trigger], c], port.loc[trigger.index[trigger]]) if trigger.any() else np.nan
        max_corr_sharpe = np.nan
        if np.isfinite(sr_i) and np.isfinite(sr_p) and abs(sr_p) > 1e-12:
            max_corr_sharpe = sr_i / sr_p
        max_corr_vol = np.nan
        if np.isfinite(vol_i) and vol_i > 1e-12 and np.isfinite(vol_p):
            max_corr_vol = vol_p / vol_i
        # Cap thresholds to interpretable correlation range, but preserve raw values too.
        mc_sh_clip = float(np.clip(max_corr_sharpe, -1, 1)) if np.isfinite(max_corr_sharpe) else np.nan
        mc_vol_clip = float(np.clip(max_corr_vol, -1, 1)) if np.isfinite(max_corr_vol) else np.nan
        rows.append({
            "Strategy": c,
            "Weight": float(w.get(c, 0.0)),
            "Sharpe": sr_i,
            "Volatility": vol_i,
            "Corr to Portfolio": corr_p,
            "Tail Corr to Portfolio": tail_corr,
            "Max Corr for Sharpe Accretion": mc_sh_clip,
            "Max Corr for Vol Reduction": mc_vol_clip,
            "Sharpe Corr Safety": mc_sh_clip - corr_p if np.isfinite(mc_sh_clip) and np.isfinite(corr_p) else np.nan,
            "Vol Corr Safety": mc_vol_clip - corr_p if np.isfinite(mc_vol_clip) and np.isfinite(corr_p) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Decision"] = np.select(
            [out["Tail Corr to Portfolio"] > 0.85, out["Sharpe Corr Safety"] < 0, out["Vol Corr Safety"] < 0],
            ["High tail-corr risk", "Not Sharpe-accretive", "Does not reduce vol"],
            default="Potentially useful",
        )
    return out.sort_values(["Decision", "Sharpe Corr Safety"], ascending=[True, False])


def pairwise_required_correlation(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise current vs required correlation threshold matrix.

    For each ordered pair i,j, the threshold is min(SR_i/SR_j, vol_j/vol_i). This is an
    approximate maximum correlation for i to be useful relative to j on Sharpe/vol terms.
    The safety matrix uses the symmetric average of ordered thresholds minus observed corr.
    """
    r = _clean(returns).fillna(0.0)
    cols = list(r.columns)
    corr = r.corr()
    req = pd.DataFrame(np.nan, index=cols, columns=cols)
    metrics = {c: pl.metrics_for_series(r[c]) for c in cols}
    for i in cols:
        for j in cols:
            if i == j:
                req.loc[i, j] = 1.0
                continue
            sr_i = metrics[i].get("Sharpe", np.nan)
            sr_j = metrics[j].get("Sharpe", np.nan)
            vol_i = metrics[i].get("Volatility", np.nan)
            vol_j = metrics[j].get("Volatility", np.nan)
            vals = []
            if np.isfinite(sr_i) and np.isfinite(sr_j) and abs(sr_j) > 1e-12:
                vals.append(sr_i / sr_j)
            if np.isfinite(vol_i) and vol_i > 1e-12 and np.isfinite(vol_j):
                vals.append(vol_j / vol_i)
            req.loc[i, j] = np.clip(np.nanmin(vals), -1, 1) if vals else np.nan
    req_sym = (req + req.T) / 2.0
    safety = req_sym - corr
    return corr, req_sym, safety


def correlation_breakpoint_curve(returns: pd.DataFrame, weights: pd.Series | Dict[str, float], levels: Optional[np.ndarray] = None) -> pd.DataFrame:
    r = _clean(returns).fillna(0.0)
    if r.empty:
        return pd.DataFrame()
    w = pl.normalize_weights(weights, list(r.columns))
    if levels is None:
        levels = np.linspace(0, 1, 21)
    rows = []
    mean = r.mean(axis=1)
    for level in levels:
        stressed = (1.0 - level) * r + level * pd.DataFrame({c: mean for c in r.columns}, index=r.index)
        p = pl.portfolio_returns(stressed, w, "Portfolio")
        m = pl.metrics_for_series(p)
        m["Correlation Stress Level"] = float(level)
        m["Avg Pairwise Corr"] = _upper_mean(stressed.corr())
        rows.append(m)
    return pd.DataFrame(rows)


def generate_random_portfolios(returns: pd.DataFrame, n: int = 4000, max_weight: float = 1.0, seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = _clean(returns).fillna(0.0)
    cols = list(r.columns)
    if not cols:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    weights = []
    rows = []
    for k in range(max(1, int(n))):
        raw = rng.dirichlet(np.ones(len(cols)))
        w = pl.normalize_weights(pd.Series(raw, index=cols), cols, max_weight)
        m = pl.portfolio_metrics(r, w, f"random_{k}")
        m["HHI"] = float((w ** 2).sum())
        rows.append(m)
        rec = {"Portfolio": f"random_{k}"}
        rec.update({c: float(w[c]) for c in cols})
        weights.append(rec)
    pts = pd.DataFrame(rows)
    wdf = pd.DataFrame(weights)
    if not pts.empty:
        pts["Efficient"] = False
        # Approximate efficient frontier by retaining max CAGR for drawdown/vol bins.
        pts = pts.sort_values("Volatility")
        best = -np.inf
        efficient_idx = []
        for idx, row in pts.iterrows():
            if row.get("CAGR", -np.inf) > best:
                efficient_idx.append(idx)
                best = row.get("CAGR", best)
        pts.loc[efficient_idx, "Efficient"] = True
    return pts, wdf


def allocation_frontier_by_return(frontier_points: pd.DataFrame, weights_df: pd.DataFrame, max_points: int = 60) -> pd.DataFrame:
    if frontier_points.empty or weights_df.empty or "Efficient" not in frontier_points.columns:
        return pd.DataFrame()
    eff = frontier_points[frontier_points["Efficient"]].sort_values("CAGR")
    if eff.empty:
        return pd.DataFrame()
    if len(eff) > max_points:
        eff = eff.iloc[np.linspace(0, len(eff) - 1, max_points).astype(int)]
    merged = weights_df.merge(eff[["Portfolio", "CAGR", "Volatility", "Max Drawdown"]], on="Portfolio", how="inner")
    strat_cols = [c for c in merged.columns if c not in {"Portfolio", "CAGR", "Volatility", "Max Drawdown"}]
    long = merged.melt(id_vars=["Portfolio", "CAGR", "Volatility", "Max Drawdown"], value_vars=strat_cols, var_name="Strategy", value_name="Weight")
    return long


def _fast_random_search(returns: pd.DataFrame, objective: str, max_weight: float = 0.5, n_samples: int = 250, seed: int = 1) -> pd.Series:
    r = _clean(returns).fillna(0.0)
    cols = list(r.columns)
    if not cols:
        return pd.Series(dtype=float)
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)
    rng = np.random.default_rng(seed)
    best_w = pd.Series(np.ones(len(cols))/len(cols), index=cols)
    best_score = -1e18
    for _ in range(int(n_samples)):
        w = pl.normalize_weights(pd.Series(rng.dirichlet(np.ones(len(cols))), index=cols), cols, max_weight)
        m = pl.portfolio_metrics(r, w, "candidate")
        if not m:
            continue
        if objective == "Max Calmar":
            score = m.get("Calmar", -1e9)
        elif objective == "Min cVaR":
            score = m.get("cVaR 95", -1e9)
        elif objective == "Robust Recommended":
            # Use Ulcer Index internally for drawdown persistence instead of a
            # direct max-drawdown-only penalty. It is not exposed as a general
            # metric column in the UI.
            p_ret = pl.portfolio_returns(r, w, "candidate")
            ui = pl.ulcer_index(p_ret)
            score = (
                1.1*m.get("CAGR", 0)
                + 0.8*m.get("Sharpe", 0)
                + 0.8*m.get("Calmar", 0)
                - 1.75*(ui if np.isfinite(ui) else 0.0)
                + 3*m.get("cVaR 95", 0)
                - 0.3*float((w**2).sum())
            )
        else:
            score = m.get("Sharpe", -1e9)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_w = w
    return best_w


def robustness_by_period(returns: pd.DataFrame, max_weight: float = 0.5, min_obs: int = 63, objectives: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = _clean(returns).fillna(0.0)
    if objectives is None:
        objectives = ["Equal Weight", "Inverse Volatility", "Max Sharpe", "Max Calmar", "Min cVaR", "Robust Recommended"]
    if r.empty:
        return pd.DataFrame(), pd.DataFrame()
    periods = r.groupby(r.index.year)
    metric_rows = []
    weight_rows = []
    for year, sub in periods:
        if len(sub) < min_obs:
            continue
        candidates = {
            "Equal Weight": pl.normalize_weights({c: 1/len(sub.columns) for c in sub.columns}, list(sub.columns), max_weight),
            "Inverse Volatility": pl.inverse_vol_weights(sub, max_weight),
            "Max Sharpe": _fast_random_search(sub, "Max Sharpe", max_weight=max_weight, n_samples=30, seed=int(year)),
            "Max Calmar": _fast_random_search(sub, "Max Calmar", max_weight=max_weight, n_samples=30, seed=int(year)+1),
            "Min cVaR": _fast_random_search(sub, "Min cVaR", max_weight=max_weight, n_samples=30, seed=int(year)+2),
            "Robust Recommended": _fast_random_search(sub, "Robust Recommended", max_weight=max_weight, n_samples=40, seed=int(year)+3),
        }
        for obj in objectives:
            if obj not in candidates:
                continue
            w = candidates[obj]
            m = pl.portfolio_metrics(sub, w, obj)
            m["Period"] = int(year)
            metric_rows.append(m)
            for s, val in w.items():
                weight_rows.append({"Period": int(year), "Portfolio": obj, "Strategy": s, "Weight": float(val)})
    return pd.DataFrame(metric_rows), pd.DataFrame(weight_rows)


def candidate_monte_carlo_comparison(
    returns: pd.DataFrame,
    candidates: Dict[str, pd.Series],
    n_sims: int = 500,
    horizon_years: int = 5,
    frequency: str = "Weekly",
    block_len: int = 8,
    seed: int = 11,
    dd_limit: float = -0.30,
) -> pd.DataFrame:
    rows = []
    for i, (name, w) in enumerate(candidates.items()):
        _, summ = pl.monte_carlo_composite(
            returns,
            w,
            n_sims=n_sims,
            horizon_years=horizon_years,
            frequency=frequency,
            block_len=block_len,
            seed=seed + i * 17,
            return_haircut=0.5,
            vol_multiplier=1.25,
            corr_stress=0.25,
            common_shock_prob=0.02,
            common_shock_severity=-0.08,
        )
        if summ.empty:
            continue
        rows.append({
            "Portfolio": name,
            "Median CAGR": float(summ["CAGR"].median()),
            "5% CAGR": float(summ["CAGR"].quantile(0.05)),
            "Median Max DD": float(summ["Max Drawdown"].median()),
            "5% Max DD": float(summ["Max Drawdown"].quantile(0.05)),
            f"P(Max DD < {dd_limit:.0%})": float((summ["Max Drawdown"] < dd_limit).mean()),
            "P(Final < 1x)": float((summ["Final Wealth Multiple"] < 1.0).mean()),
            "Median Final Wealth": float(summ["Final Wealth Multiple"].median()),
        })
    return pd.DataFrame(rows).set_index("Portfolio") if rows else pd.DataFrame()


def final_decision_table(metrics: pd.DataFrame, corr_req: pd.DataFrame, stress_curve: pd.DataFrame, mc_comp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build a readable final-decision component table for current candidate metrics.

    This is intentionally transparent: it uses percentile ranks and exposes each component.
    """
    if metrics.empty:
        return pd.DataFrame()
    df = metrics.copy()
    def rank(s, high_good=True):
        return s.rank(pct=True, ascending=high_good).fillna(0.5)
    out = pd.DataFrame(index=df.index)
    out["Return Score"] = rank(df.get("CAGR", pd.Series(index=df.index, dtype=float)), True)
    out["Risk Score"] = (rank(df.get("Max Drawdown", pd.Series(index=df.index, dtype=float)), True) + rank(df.get("cVaR 95", pd.Series(index=df.index, dtype=float)), True)) / 2
    out["Risk-Adjusted Score"] = (rank(df.get("Sharpe", pd.Series(index=df.index, dtype=float)), True) + rank(df.get("Calmar", pd.Series(index=df.index, dtype=float)), True)) / 2
    # Higher HHI means more concentration, so the penalty must be larger for
    # higher HHI.  The previous descending rank inverted this penalty.
    out["Concentration Penalty"] = rank(df.get("Concentration HHI", pd.Series(index=df.index, dtype=float)), True) * 0.35 if "Concentration HHI" in df else 0.0
    if mc_comp is not None and not mc_comp.empty:
        mc = mc_comp.reindex(out.index)
        out["MC Downside Score"] = (rank(mc.get("5% CAGR", pd.Series(index=out.index, dtype=float)), True) + rank(mc.get("5% Max DD", pd.Series(index=out.index, dtype=float)), True)) / 2
    else:
        out["MC Downside Score"] = 0.5
    # Active correlation requirements are strategy-level not candidate-level; apply mean safety as global backdrop.
    mean_safety = float(corr_req["Sharpe Corr Safety"].mean()) if corr_req is not None and not corr_req.empty and "Sharpe Corr Safety" in corr_req else 0.0
    out["Correlation Safety Backdrop"] = max(0.0, min(1.0, 0.5 + mean_safety / 2.0))
    out["Final Score"] = 1.1*out["Return Score"] + 1.25*out["Risk Score"] + 1.1*out["Risk-Adjusted Score"] + 1.0*out["MC Downside Score"] + 0.7*out["Correlation Safety Backdrop"] - out["Concentration Penalty"]
    out["Decision"] = pd.cut(out["Final Score"], [-np.inf, out["Final Score"].quantile(.4), out["Final Score"].quantile(.75), np.inf], labels=["Reject / research only", "Viable", "Recommended shortlist"])
    return out.sort_values("Final Score", ascending=False)


def excel_workbook_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Create a polished multi-sheet research workbook.

    The workbook is meant to be read, not just stored: frozen panes, filters,
    finance-aware formats, readable widths, and conditional formatting are applied.
    """
    import io
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#17365D", "font_color": "white", "border": 1, "align": "center"})
        text_fmt = workbook.add_format({"border": 1})
        num_fmt = workbook.add_format({"num_format": "#,##0.00", "border": 1})
        int_fmt = workbook.add_format({"num_format": "#,##0", "border": 1})
        pct_fmt = workbook.add_format({"num_format": "0.00%", "border": 1})
        ratio_fmt = workbook.add_format({"num_format": "0.00", "border": 1})
        date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd", "border": 1})
        for name, df in sheets.items():
            safe = str(name)[:31].replace("/", "-").replace("\\", "-") or "Sheet"
            if df is None or df.empty:
                pd.DataFrame({"Note": ["No data available"]}).to_excel(writer, sheet_name=safe, index=False)
                ws = writer.sheets[safe]
                ws.set_column(0, 0, 28)
                continue
            out = df.copy()
            write_index = not isinstance(out.index, pd.RangeIndex)
            out.to_excel(writer, sheet_name=safe, index=write_index)
            ws = writer.sheets[safe]
            nrows, ncols = out.shape
            index_cols = 1 if write_index else 0
            total_cols = ncols + index_cols
            ws.freeze_panes(1, index_cols)
            ws.autofilter(0, 0, nrows, max(total_cols - 1, 0))
            headers = ([out.index.name or "Index"] if write_index else []) + list(out.columns)
            for j, h in enumerate(headers):
                ws.write(0, j, str(h), header_fmt)
            for j, col_name in enumerate(headers):
                data_col = pd.Series(out.index) if write_index and j == 0 else out.iloc[:, j - index_cols]
                low = str(col_name).lower()
                width = min(max(len(str(col_name)) + 2, 12), 42)
                try:
                    sample_width = int(min(max(data_col.astype(str).map(len).quantile(0.90) + 2, width), 45))
                    width = max(width, sample_width)
                except Exception:
                    pass
                if any(k in low for k in ["return", "cagr", "vol", "drawdown", "dd", "weight", "var", "cvar", "win", "missing", "prob", "p(", "corr", "beta", "alpha", "contribution", "exposure"]):
                    fmt = pct_fmt
                elif any(k in low for k in ["sharpe", "sortino", "calmar", "score", "hhi", "multiple"]):
                    fmt = ratio_fmt
                elif "date" in low or low in {"start", "end"}:
                    fmt = date_fmt
                elif hasattr(data_col, "dtype") and data_col.dtype.kind in "iu":
                    fmt = int_fmt
                elif hasattr(data_col, "dtype") and data_col.dtype.kind in "fc":
                    fmt = num_fmt
                else:
                    fmt = text_fmt
                ws.set_column(j, j, width, fmt)
            for j, h in enumerate(headers):
                low = str(h).lower()
                if nrows <= 0:
                    continue
                if any(k in low for k in ["score", "safety", "cagr", "sharpe", "calmar"]):
                    ws.conditional_format(1, j, nrows, j, {"type": "3_color_scale"})
                elif any(k in low for k in ["drawdown", "cvar", "var", "loss"]):
                    ws.conditional_format(1, j, nrows, j, {"type": "3_color_scale", "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B"})
            ws.set_zoom(90)
    return bio.getvalue()

