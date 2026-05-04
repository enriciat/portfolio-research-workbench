from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

TRADING_DAYS = 252
RISK_FREE_RATE = 0.0


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "strategy"


def _file_bytes(file_meta: Dict[str, Any]) -> bytes:
    """Read bytes from either in-memory metadata or a filesystem path."""
    if "bytes" in file_meta and file_meta["bytes"] is not None:
        return file_meta["bytes"]
    full_path = file_meta.get("full_path") or file_meta.get("abs_path")
    if full_path:
        return Path(full_path).read_bytes()
    raise FileNotFoundError(f"No bytes or full_path for {file_meta.get('path')}")


def parse_percent_weight(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    s = str(value).strip().replace("%", "")
    if not s:
        return 0.0
    try:
        x = float(s)
    except Exception:
        return 0.0
    return x / 100.0 if abs(x) > 1.0 else x


def load_returns_from_outputs(output_files: List[Dict[str, Any]], index: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load strategy and benchmark returns from generated .equity.csv files.

    Missing data is preserved as NaN. Alignment/filling is a UI decision, not a parser decision.
    """
    path_to_file = {f["path"]: f for f in output_files}
    items = index.get("items", []) if isinstance(index, dict) else []
    if not items:
        items = []
        for f in output_files:
            if f["path"].endswith(".equity.csv"):
                stem = f["path"].replace(".equity.csv", "")
                items.append({"name": stem, "equity_csv": f["path"]})

    strategy_series: Dict[str, pd.Series] = {}
    benchmark_series: Dict[str, pd.Series] = {}
    equity_path_map: Dict[str, str] = {}
    used_names: Dict[str, int] = {}

    for item in items:
        eq_path = item.get("equity_csv")
        if not eq_path or eq_path not in path_to_file:
            continue
        try:
            raw = _file_bytes(path_to_file[eq_path])
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            continue
        if "Date" not in df.columns or "Strategy Equity" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        strat_eq = pd.to_numeric(df["Strategy Equity"], errors="coerce").dropna()
        if len(strat_eq) < 3:
            continue
        name = str(item.get("name") or eq_path.replace(".equity.csv", ""))
        base_name = name
        if name in used_names:
            used_names[name] += 1
            name = f"{base_name} ({used_names[base_name]})"
        else:
            used_names[name] = 1
        rets = strat_eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        strategy_series[name] = rets
        equity_path_map[name] = eq_path
        if "Benchmark Equity" in df.columns:
            bench_eq = pd.to_numeric(df["Benchmark Equity"], errors="coerce").dropna()
            if len(bench_eq) >= 3:
                bret = bench_eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                benchmark_series[f"Benchmark for {name}"] = bret

    return pd.DataFrame(strategy_series).sort_index(), pd.DataFrame(benchmark_series).sort_index(), equity_path_map


def load_latest_allocations(output_files: List[Dict[str, Any]], index: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    path_to_file = {f["path"]: f for f in output_files}
    items = index.get("items", []) if isinstance(index, dict) else []
    if not items:
        items = []
        for f in output_files:
            if f["path"].endswith(".allocation.csv"):
                stem = f["path"].replace(".allocation.csv", "")
                items.append({"name": stem, "allocation_csv": f["path"]})

    out: Dict[str, pd.DataFrame] = {}
    for item in items:
        path = item.get("allocation_csv")
        name = str(item.get("name") or path or "strategy")
        if not path or path not in path_to_file:
            continue
        try:
            df = pd.read_csv(io.BytesIO(_file_bytes(path_to_file[path])))
        except Exception:
            continue
        if df.empty or "Assets" not in df.columns or "Weights" not in df.columns:
            continue
        latest = df.iloc[-1]
        assets = [x.strip().upper() for x in str(latest.get("Assets", "")).split(",") if x.strip()]
        weights = [parse_percent_weight(x) for x in str(latest.get("Weights", "")).split(",") if str(x).strip()]
        rows = [{"Ticker": a, "Weight": w, "Date": latest.get("Date", "")} for a, w in zip(assets, weights)]
        out[name] = pd.DataFrame(rows).sort_values("Weight", ascending=False) if rows else pd.DataFrame(columns=["Ticker", "Weight", "Date"])
    return out


def align_returns(returns: pd.DataFrame, mode: str) -> pd.DataFrame:
    if returns.empty:
        return returns
    if mode.startswith("Common"):
        return returns.dropna(how="any")
    if mode.startswith("Full history with cash"):
        return returns.fillna(0.0)
    # Inception mode: preserve NaNs before a strategy exists; fill isolated missing after inception with 0.
    out = returns.copy()
    for col in out.columns:
        first = out[col].first_valid_index()
        if first is not None:
            out.loc[first:, col] = out.loc[first:, col].fillna(0.0)
    return out


def composite_returns(returns: pd.DataFrame, weights: Dict[str, float], name: str = "Composite Portfolio") -> pd.Series:
    cols = [c for c, w in weights.items() if c in returns.columns and float(w) != 0]
    if not cols:
        return pd.Series(dtype=float, name=name)
    w = pd.Series({c: float(weights[c]) for c in cols}, dtype=float)
    if w.sum() == 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()
    r = returns[cols].fillna(0.0).dot(w)
    r.name = name
    return r


def drawdowns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    wealth = (1 + returns.fillna(0)).cumprod()
    peaks = wealth.cummax()
    return wealth / peaks - 1


def _years_by_calendar(r: pd.Series) -> float:
    if len(r) < 2:
        return max(len(r) / TRADING_DAYS, 1 / TRADING_DAYS)
    days = max((r.index.max() - r.index.min()).days, 1)
    return max(days / 365.25, 1 / 365.25)


def downside_deviation(r: pd.Series, target: float = 0.0) -> float:
    down = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(down ** 2)) * np.sqrt(TRADING_DAYS)) if len(r) else np.nan


def perf_summary(returns: pd.DataFrame, risk_free_rate: float = RISK_FREE_RATE) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) < 3:
            continue
        total = (1 + r).prod() - 1
        years = _years_by_calendar(r)
        cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
        vol = r.std() * np.sqrt(TRADING_DAYS)
        ann_mean = r.mean() * TRADING_DAYS
        sharpe = (ann_mean - risk_free_rate) / vol if vol > 0 else np.nan
        ddev = downside_deviation(r)
        sortino = (ann_mean - risk_free_rate) / ddev if ddev > 0 else np.nan
        dd = drawdowns(r)
        max_dd = float(dd.min()) if len(dd) else np.nan
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
        var95 = float(r.quantile(0.05))
        cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95
        monthly = (1 + r).resample("ME").prod() - 1
        yearly = (1 + r).resample("YE").prod() - 1
        rows.append({
            "Strategy": col,
            "Start": r.index.min().date(),
            "End": r.index.max().date(),
            "Observations": len(r),
            "Calendar Years": years,
            "Missing %": float(returns[col].isna().mean()),
            "Total Return": total,
            "CAGR": cagr,
            "Volatility": vol,
            "Max Drawdown": max_dd,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "Worst Day": float(r.min()),
            "Worst Month": float(monthly.min()) if len(monthly) else np.nan,
            "Worst Year": float(yearly.min()) if len(yearly) else np.nan,
            "Win Day %": float((r > 0).mean()),
            "Win Month %": float((monthly > 0).mean()) if len(monthly) else np.nan,
            "Win Year %": float((yearly > 0).mean()) if len(yearly) else np.nan,
            "VaR 95": var95,
            "cVaR 95": cvar95,
            "Skew": float(r.skew()),
            "Kurtosis": float(r.kurtosis()),
        })
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson", mask_insignificant: bool = False, p_threshold: float = 0.05) -> pd.DataFrame:
    r = returns.dropna(how="all")
    corr = r.corr(method=method)
    if not mask_insignificant or r.shape[1] < 2:
        return corr
    pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if i >= j:
                continue
            valid = r[[c1, c2]].dropna()
            if len(valid) < 5:
                continue
            try:
                if method == "spearman":
                    _rho, p = stats.spearmanr(valid[c1], valid[c2])
                elif method == "kendall":
                    _rho, p = stats.kendalltau(valid[c1], valid[c2])
                else:
                    _rho, p = stats.pearsonr(valid[c1], valid[c2])
                pvals.loc[c1, c2] = pvals.loc[c2, c1] = p
            except Exception:
                pass
    return corr.mask(pvals > p_threshold)


def tail_correlation(returns: pd.DataFrame, trigger: Optional[str] = None, threshold: float = -0.05) -> Tuple[pd.DataFrame, int]:
    if returns.empty:
        return pd.DataFrame(), 0
    if trigger is None or trigger not in returns.columns:
        trigger = returns.columns[0]
    dd = drawdowns(returns[[trigger]]).iloc[:, 0]
    stress_mask = dd < threshold
    data = returns.loc[stress_mask]
    if len(data) < 8:
        return pd.DataFrame(), len(data)
    return data.corr(), len(data)


def diversification_metrics(returns: pd.DataFrame) -> Dict[str, float]:
    if returns.shape[1] < 2:
        return {"Avg Pairwise Corr": 1.0, "Diversification Ratio": 1.0, "Max Pairwise Corr": 1.0}
    corr = returns.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    avg_corr = float(upper.mean()) if len(upper) else np.nan
    max_corr = float(upper.max()) if len(upper) else np.nan
    n = returns.shape[1]
    w = np.ones(n) / n
    weighted_vol_sum = float(np.sum(returns.std() * w))
    cov = returns.cov()
    port_vol = float(np.sqrt(w.T @ cov.values @ w)) if not cov.empty else np.nan
    div_ratio = weighted_vol_sum / port_vol if port_vol and port_vol > 0 else np.nan
    return {"Avg Pairwise Corr": avg_corr, "Diversification Ratio": div_ratio, "Max Pairwise Corr": max_corr}


def similarity_pairs(returns: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    corr = returns.corr()
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(val) >= threshold:
                rows.append({"Strategy A": a, "Strategy B": b, "Correlation": val})
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False) if rows else pd.DataFrame()


def cluster_linkage(returns: pd.DataFrame):
    corr = returns.corr().fillna(0)
    dist = np.sqrt(2 * (1 - corr)).clip(lower=0)
    dist_arr = dist.to_numpy(copy=True)
    # Copy before mutating; some Pandas/NumPy versions expose read-only views.
    dist_arr[np.diag_indices_from(dist_arr)] = 0.0
    return hierarchy.linkage(squareform(dist_arr, checks=False), method="ward")


def rolling_corr(returns: pd.DataFrame, left: str, right: str, window: int = 63) -> pd.Series:
    return returns[left].rolling(window).corr(returns[right])


def rolling_metrics(returns: pd.DataFrame, window: int = 63) -> Dict[str, pd.DataFrame]:
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    roll_vol = roll_std * np.sqrt(TRADING_DAYS)
    roll_sharpe = (roll_mean * TRADING_DAYS) / roll_vol.replace(0, np.nan)
    roll_cagr = (1 + returns).rolling(window).apply(np.prod, raw=True) ** (TRADING_DAYS / window) - 1
    roll_dd = returns.rolling(window).apply(lambda x: float(drawdowns(pd.Series(x)).min()), raw=False)
    return {"Rolling Volatility": roll_vol, "Rolling Sharpe": roll_sharpe, "Rolling CAGR": roll_cagr, "Rolling Max Drawdown": roll_dd}


def beta_to_benchmark(strategy: pd.Series, benchmark: pd.Series, window: Optional[int] = None) -> pd.Series | float:
    data = pd.concat([strategy, benchmark], axis=1).dropna()
    if data.shape[1] < 2 or len(data) < 5:
        return np.nan if window is None else pd.Series(dtype=float)
    s = data.iloc[:, 0]
    b = data.iloc[:, 1]
    if window is None:
        var = b.var()
        return float(s.cov(b) / var) if var != 0 else np.nan
    return s.rolling(window).cov(b) / b.rolling(window).var()


def factor_regression(strategy: pd.Series, factors: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    data = pd.concat([strategy.rename("strategy"), factors], axis=1).dropna()
    if len(data) < max(10, factors.shape[1] + 3):
        return pd.DataFrame(), {}
    y = data["strategy"].values
    X = data[factors.columns].values
    Xc = np.column_stack([np.ones(len(X)), X])
    try:
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        yhat = Xc @ beta
        resid = y - yhat
        n, p = Xc.shape
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst if sst > 0 else np.nan
        sigma2 = sse / max(n - p, 1)
        cov_beta = sigma2 * np.linalg.pinv(Xc.T @ Xc)
        se = np.sqrt(np.diag(cov_beta))
        t_stats = beta / np.where(se == 0, np.nan, se)
        cols = ["Alpha"] + list(factors.columns)
        res = pd.DataFrame({"Coefficient": beta, "T-Stat": t_stats}, index=cols)
        stats_dict = {"R2": r2, "Residual Vol": float(np.std(resid) * np.sqrt(TRADING_DAYS)), "Annual Alpha": float(beta[0] * TRADING_DAYS)}
        return res, stats_dict
    except Exception:
        return pd.DataFrame(), {}


def latest_allocation_table(allocations: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for strat, df in allocations.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            rows.append({"Strategy": strat, "Ticker": row.get("Ticker"), "Weight": row.get("Weight"), "Date": row.get("Date")})
    return pd.DataFrame(rows)


def concentration_stats(alloc_df: pd.DataFrame) -> Dict[str, float]:
    if alloc_df.empty or "Weight" not in alloc_df.columns:
        return {"Top 1": np.nan, "Top 3": np.nan, "HHI": np.nan}
    w = alloc_df["Weight"].astype(float).clip(lower=0).sort_values(ascending=False)
    total = w.sum()
    if total > 0:
        w = w / total
    return {"Top 1": float(w.head(1).sum()), "Top 3": float(w.head(3).sum()), "HHI": float((w ** 2).sum())}


def holdings_risk_contribution(weights: pd.Series, asset_returns: pd.DataFrame) -> pd.Series:
    weights = weights.dropna().astype(float)
    cols = [c for c in weights.index if c in asset_returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    w = weights.loc[cols]
    if w.sum() != 0:
        w = w / w.sum()
    r = asset_returns[cols].dropna(how="all").fillna(0)
    if len(r) < 10:
        return pd.Series(dtype=float)
    cov = r.cov()
    port_var = float(w.values.T @ cov.values @ w.values)
    if port_var <= 0:
        return pd.Series(dtype=float)
    contrib = w.values * (cov.values @ w.values) / port_var
    return pd.Series(contrib, index=cols).sort_values(ascending=False)


def top_drawdowns(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    r = series.dropna()
    if r.empty:
        return pd.DataFrame()
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    rows = []
    in_dd = False
    start = trough = end = None
    min_dd = 0.0
    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd = True
            start = date
            trough = date
            min_dd = float(val)
        elif in_dd:
            if val < min_dd:
                min_dd = float(val)
                trough = date
            if val >= -1e-12:
                end = date
                rows.append({"Start": start, "Trough": trough, "Recovery": end, "Depth": min_dd, "Duration Days": (end - start).days})
                in_dd = False
    if in_dd and start is not None:
        rows.append({"Start": start, "Trough": trough, "Recovery": pd.NaT, "Depth": min_dd, "Duration Days": (r.index[-1] - start).days})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Depth").head(top_n)


def monthly_returns_table(series: pd.Series) -> pd.DataFrame:
    r = series.dropna()
    if r.empty:
        return pd.DataFrame()
    monthly = (1 + r).resample("ME").prod() - 1
    df = monthly.to_frame("Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month_name().str[:3]
    pivot = df.pivot(index="Year", columns="Month", values="Return")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot.reindex(columns=order)


def yearly_returns(series: pd.Series) -> pd.Series:
    r = series.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    y = (1 + r).resample("YE").prod() - 1
    y.index = y.index.year
    return y


def parse_factor_csv(uploaded) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        raw = pd.read_csv(uploaded)
        if raw.empty:
            return pd.DataFrame(), "CSV is empty."
        date_col = "Date" if "Date" in raw.columns else raw.columns[0]
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
        num = raw.apply(pd.to_numeric, errors="coerce")
        if num.empty:
            return pd.DataFrame(), "No numeric factor columns found."
        # If values look like price levels, convert to returns; otherwise treat as returns.
        if num.abs().median(numeric_only=True).median() > 2:
            num = num.pct_change()
        return num.replace([np.inf, -np.inf], np.nan), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def generate_analytics_addendum_html(
    returns: pd.DataFrame,
    allocations: Dict[str, pd.DataFrame],
    selected_strategy: Optional[str] = None,
    title: str = "Advanced Analytics Addendum",
) -> str:
    summary = perf_summary(returns)
    corr = returns.corr() if not returns.empty else pd.DataFrame()
    tail_corr, n_tail = tail_correlation(returns, trigger=selected_strategy or (returns.columns[0] if len(returns.columns) else None))
    div = diversification_metrics(returns) if not returns.empty else {}
    selected_strategy = selected_strategy or (returns.columns[0] if len(returns.columns) else None)

    html = [
        "<html><head><meta charset='utf-8'><title>Analytics Addendum</title>",
        "<script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:30px;color:#222}.kpi{display:inline-block;background:#f7f7f7;border:1px solid #ddd;border-radius:8px;padding:12px;margin:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:6px;text-align:right}th{text-align:center;background:#f0f0f0}h1,h2{color:#123}.warn{background:#fff6db;border:1px solid #e5c56e;padding:10px;border-radius:6px}</style>",
        "</head><body>",
        f"<h1>{title}</h1>",
        "<p>This addendum is generated from the backtest equity/allocation outputs. It focuses on performance, drawdown, correlation, clustering, and exposure.</p>",
    ]
    if div:
        for k, v in div.items():
            html.append(f"<div class='kpi'><b>{k}</b><br>{v:.3f}</div>")
    if not summary.empty:
        display = summary.copy()
        for c in display.columns:
            if display[c].dtype.kind in "fc":
                display[c] = display[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        html.append("<h2>Performance Summary</h2>")
        html.append(display.to_html(escape=False))
        # Visual companion: CAGR vs Max Drawdown scatter.
        fig_scatter = px.scatter(summary.reset_index(), x="Max Drawdown", y="CAGR", size="Volatility", hover_name="Strategy", title="Risk/Return Map")
        html.append(fig_scatter.to_html(full_html=False, include_plotlyjs=False))
    if not returns.empty:
        fig_eq = px.line((1 + returns.fillna(0)).cumprod(), title="Equity Curves", log_y=True)
        html.append("<h2>Equity Curves</h2>")
        html.append(fig_eq.to_html(full_html=False, include_plotlyjs=False))
        fig_dd = px.line(drawdowns(returns), title="Drawdowns")
        html.append("<h2>Drawdowns</h2>")
        html.append(fig_dd.to_html(full_html=False, include_plotlyjs=False))
        if returns.shape[1] > 1:
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Matrix")
            html.append("<h2>Correlation Matrix</h2>")
            html.append(fig_corr.to_html(full_html=False, include_plotlyjs=False))
            if not tail_corr.empty:
                fig_tail = px.imshow(tail_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=f"Tail Correlation ({n_tail} stress observations)")
                html.append("<h2>Tail Correlation</h2>")
                html.append(fig_tail.to_html(full_html=False, include_plotlyjs=False))
        if selected_strategy in returns.columns:
            tdd = top_drawdowns(returns[selected_strategy])
            if not tdd.empty:
                html.append(f"<h2>Top Drawdowns: {selected_strategy}</h2>")
                show = tdd.copy()
                show["Depth"] = show["Depth"].map(lambda x: f"{x:.2%}")
                html.append(show.to_html(index=False, escape=False))
            m = monthly_returns_table(returns[selected_strategy])
            if not m.empty:
                fig_month = px.imshow(m, text_auto=".1%", color_continuous_scale="RdYlGn", aspect="auto", title=f"Monthly Return Heatmap: {selected_strategy}")
                html.append("<h2>Monthly Return Heatmap</h2>")
                html.append(fig_month.to_html(full_html=False, include_plotlyjs=False))
            y = yearly_returns(returns[selected_strategy])
            if not y.empty:
                fig_year = px.bar(y, title=f"Yearly Returns: {selected_strategy}")
                html.append("<h2>Yearly Returns</h2>")
                html.append(fig_year.to_html(full_html=False, include_plotlyjs=False))
    alloc_table = latest_allocation_table(allocations)
    if not alloc_table.empty:
        html.append("<h2>Latest Allocations</h2>")
        show = alloc_table.copy()
        show["Weight"] = show["Weight"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
        html.append(show.to_html(index=False, escape=False))
        top = alloc_table.groupby("Ticker")["Weight"].sum().sort_values(ascending=False).head(25).reset_index()
        fig_alloc = px.bar(top, x="Ticker", y="Weight", title="Combined Top Holdings")
        html.append(fig_alloc.to_html(full_html=False, include_plotlyjs=False))
    html.append("</body></html>")
    return "\n".join(html)
