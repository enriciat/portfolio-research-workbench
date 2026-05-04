from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

TRADING_DAYS = 252


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns.copy().replace([np.inf, -np.inf], np.nan)
    return r.dropna(how="all")


def normalize_weights(weights: pd.Series | Dict[str, float], columns: Optional[List[str]] = None, max_weight: float = 1.0) -> pd.Series:
    if isinstance(weights, dict):
        w = pd.Series(weights, dtype=float)
    else:
        w = weights.astype(float).copy()
    if columns is not None:
        w = w.reindex(columns).fillna(0.0)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if max_weight < 1.0:
        w = w.clip(upper=max_weight)
    if float(w.sum()) <= 0:
        if len(w) == 0:
            return w
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()
    # Clip once more and renormalize for max-weight constraints.
    if max_weight < 1.0:
        for _ in range(50):
            over = w > max_weight + 1e-12
            if not over.any():
                break
            fixed = w[over].clip(upper=max_weight)
            remaining = 1.0 - fixed.sum()
            rest = w[~over]
            if len(rest) == 0 or remaining <= 0:
                w = fixed.reindex(w.index).fillna(0.0)
                break
            rest = rest / rest.sum() * remaining if rest.sum() > 0 else pd.Series(remaining / len(rest), index=rest.index)
            w = pd.concat([fixed, rest]).reindex(w.index).fillna(0.0)
    return w / w.sum() if w.sum() > 0 else w


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | Dict[str, float], name: str = "Portfolio") -> pd.Series:
    r = clean_returns(returns)
    w = normalize_weights(weights, list(r.columns))
    out = r.fillna(0.0).dot(w)
    out.name = name
    return out


def drawdowns(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    wealth = (1.0 + s).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def ulcer_index(series: pd.Series) -> float:
    """Internal RMS drawdown penalty used by Robust Recommended.

    The value is intentionally not exposed as a general table metric; it is used
    only to make the Robust Recommended objective account for drawdown depth and
    persistence rather than relying on a single worst point.
    """
    dd = drawdowns(series)
    if dd.empty:
        return np.nan
    return float(np.sqrt(np.nanmean(np.square(dd.values))))


def cvar(series: pd.Series, alpha: float = 0.05) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    q = s.quantile(alpha)
    tail = s[s <= q]
    return float(tail.mean()) if len(tail) else float(q)


def downside_dev(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    down = np.minimum(s, 0.0)
    return float(np.sqrt(np.mean(down * down)) * np.sqrt(TRADING_DAYS))


def metrics_for_series(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    if len(s) < 3:
        return {}
    total = float((1 + s).prod() - 1)
    years = max((s.index.max() - s.index.min()).days / 365.25, len(s) / TRADING_DAYS, 1 / TRADING_DAYS)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(s.std() * np.sqrt(TRADING_DAYS))
    ann = float(s.mean() * TRADING_DAYS)
    dd = drawdowns(s)
    max_dd = float(dd.min()) if len(dd) else np.nan
    cv = cvar(s)
    sharpe = ann / vol if vol > 0 else np.nan
    ddv = downside_dev(s)
    sortino = ann / ddv if ddv > 0 else np.nan
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "cVaR 95": cv,
        "Total Return": total,
        "Worst Day": float(s.min()),
        "Positive Days %": float((s > 0).mean()),
    }


def portfolio_metrics(returns: pd.DataFrame, weights: pd.Series | Dict[str, float], name: str = "Portfolio") -> Dict[str, Any]:
    pr = portfolio_returns(returns, weights, name)
    m = metrics_for_series(pr)
    m["Portfolio"] = name
    return m


def inverse_vol_weights(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    vol = clean_returns(returns).std().replace(0, np.nan)
    inv = 1.0 / vol
    return normalize_weights(inv.replace([np.inf, -np.inf], np.nan).fillna(0.0), list(returns.columns), max_weight)


def inverse_drawdown_weights(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    vals = {}
    for c in returns.columns:
        md = abs(float(drawdowns(returns[c]).min())) if returns[c].dropna().size else np.nan
        vals[c] = 1.0 / md if md and np.isfinite(md) and md > 0 else 0.0
    return normalize_weights(vals, list(returns.columns), max_weight)


def min_variance_weights(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    r = clean_returns(returns).fillna(0.0)
    n = r.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    cov = r.cov().values
    if minimize is None:
        return inverse_vol_weights(r, max_weight)
    bounds = [(0.0, max_weight) for _ in range(n)]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    def obj(w): return float(w.T @ cov @ w)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 600, "ftol": 1e-10})
    if not res.success:
        return inverse_vol_weights(r, max_weight)
    return normalize_weights(pd.Series(res.x, index=r.columns), list(r.columns), max_weight)


def diversification_ratio_weights(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    r = clean_returns(returns).fillna(0.0)
    n = r.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    cov = r.cov().values
    vols = r.std().values
    if minimize is None:
        return inverse_vol_weights(r, max_weight)
    bounds = [(0.0, max_weight) for _ in range(n)]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(n) / n
    def obj(w):
        port_vol = np.sqrt(max(w.T @ cov @ w, 1e-16))
        return -float(np.dot(w, vols) / port_vol)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 600})
    if not res.success:
        return inverse_vol_weights(r, max_weight)
    return normalize_weights(pd.Series(res.x, index=r.columns), list(r.columns), max_weight)


def random_weight_search(
    returns: pd.DataFrame,
    objective: str,
    max_weight: float = 1.0,
    n_samples: int = 600,
    seed: int = 42,
    tail_corr_penalty: float = 0.0,
    concentration_penalty: float = 0.0,
) -> pd.Series:
    """Fast random-search optimizer for portfolio candidates.

    This intentionally uses a vectorized metric approximation so it remains responsive
    in Streamlit. More expensive exact metrics are computed after the candidate is
    selected.
    """
    r = clean_returns(returns).fillna(0.0)
    cols = list(r.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=cols)
    rng = np.random.default_rng(seed)
    deterministic = [
        np.ones(n) / n,
        inverse_vol_weights(r, max_weight).values,
        inverse_drawdown_weights(r, max_weight).values,
        min_variance_weights(r, max_weight).values,
    ]
    rand = rng.dirichlet(np.ones(n), size=max(1, int(n_samples)))
    if max_weight < 1.0:
        rand = np.vstack([normalize_weights(pd.Series(w, index=cols), cols, max_weight).values for w in rand])
    W = np.vstack(deterministic + [rand]) if rand.ndim == 2 else np.vstack(deterministic + [rand.reshape(1, -1)])
    W = np.vstack([normalize_weights(pd.Series(w, index=cols), cols, max_weight).values for w in W])
    R = r.values
    P = R @ W.T
    years = max((r.index.max() - r.index.min()).days / 365.25, len(r) / TRADING_DAYS, 1 / TRADING_DAYS)
    wealth = np.cumprod(1.0 + P, axis=0)
    final = wealth[-1, :]
    cagr = np.where(final > 0, final ** (1.0 / years) - 1.0, -1.0)
    vol = np.nanstd(P, axis=0) * np.sqrt(TRADING_DAYS)
    ann = np.nanmean(P, axis=0) * TRADING_DAYS
    sharpe = np.divide(ann, vol, out=np.full_like(ann, np.nan), where=vol > 0)
    downside = np.minimum(P, 0.0)
    ddev = np.sqrt(np.nanmean(downside * downside, axis=0)) * np.sqrt(TRADING_DAYS)
    sortino = np.divide(ann, ddev, out=np.full_like(ann, np.nan), where=ddev > 0)
    peak = np.maximum.accumulate(wealth, axis=0)
    dd = wealth / np.maximum(peak, 1e-16) - 1.0
    maxdd = np.nanmin(dd, axis=0)
    ulcer = np.sqrt(np.nanmean(dd * dd, axis=0))
    calmar = np.divide(cagr, np.abs(maxdd), out=np.full_like(cagr, np.nan), where=maxdd < 0)
    q = np.nanquantile(P, 0.05, axis=0)
    cvars = np.array([np.nanmean(P[:, j][P[:, j] <= q[j]]) if np.any(P[:, j] <= q[j]) else q[j] for j in range(P.shape[1])])

    if objective == "max_sharpe":
        score = sharpe
    elif objective == "max_sortino":
        score = sortino
    elif objective == "max_calmar":
        score = calmar
    elif objective == "min_drawdown":
        score = -np.abs(maxdd)
    elif objective == "min_cvar":
        score = cvars
    elif objective == "robust":
        # Robust Recommended: blend upside/risk-adjusted return with left-tail
        # risk, concentration, and drawdown persistence.  The direct drawdown
        # penalty is Ulcer Index, not only max drawdown, so long underwater
        # periods are penalized even if their single worst point is not extreme.
        score = (
            1.25 * np.nan_to_num(cagr)
            + 0.60 * np.nan_to_num(sharpe)
            + 0.35 * np.nan_to_num(calmar)
            - 1.75 * np.nan_to_num(ulcer)
            + 4.00 * np.nan_to_num(cvars)
        )
    else:
        score = sharpe
    corr = r.corr().fillna(0.0).clip(lower=0.0).to_numpy(copy=True)
    # Streamlit Cloud may run a NumPy/Pandas combination where .values returns
    # a read-only view; copy explicitly before mutating the diagonal.
    corr[np.diag_indices_from(corr)] = 0.0
    weighted_corr_penalty = np.einsum("ij,jk,ik->i", W, corr, W)
    score = np.nan_to_num(score, nan=-1e9, neginf=-1e9, posinf=1e9)
    score -= concentration_penalty * np.sum(W * W, axis=1)
    score -= tail_corr_penalty * weighted_corr_penalty
    best_idx = int(np.argmax(score))
    return normalize_weights(pd.Series(W[best_idx], index=cols), cols, max_weight)


def tail_risk_parity_weights(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    r = clean_returns(returns)
    vals = {}
    for c in r.columns:
        cv = abs(cvar(r[c]))
        md = abs(float(drawdowns(r[c]).min())) if len(r[c].dropna()) else np.nan
        risk = np.nanmean([cv, md / 20 if np.isfinite(md) else np.nan])
        vals[c] = 1.0 / risk if risk and np.isfinite(risk) and risk > 0 else 0.0
    return normalize_weights(vals, list(r.columns), max_weight)


def candidate_portfolios(returns: pd.DataFrame, max_weight: float = 1.0, n_samples: int = 600, seed: int = 42) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    r = clean_returns(returns)
    cols = list(r.columns)
    candidates: Dict[str, pd.Series] = {}
    if not cols:
        return candidates, pd.DataFrame()
    candidates["Equal Weight"] = normalize_weights({c: 1 / len(cols) for c in cols}, cols, max_weight)
    candidates["Inverse Volatility"] = inverse_vol_weights(r, max_weight)
    candidates["Inverse Drawdown"] = inverse_drawdown_weights(r, max_weight)
    candidates["Minimum Variance"] = min_variance_weights(r, max_weight)
    candidates["Max Diversification"] = diversification_ratio_weights(r, max_weight)
    candidates["Max Sharpe"] = random_weight_search(r, "max_sharpe", max_weight, n_samples=n_samples, seed=seed)
    candidates["Max Sortino"] = random_weight_search(r, "max_sortino", max_weight, n_samples=n_samples, seed=seed + 1)
    candidates["Max Calmar"] = random_weight_search(r, "max_calmar", max_weight, n_samples=n_samples, seed=seed + 2)
    candidates["Min Drawdown"] = random_weight_search(r, "min_drawdown", max_weight, n_samples=n_samples, seed=seed + 3)
    candidates["Min cVaR"] = random_weight_search(r, "min_cvar", max_weight, n_samples=n_samples, seed=seed + 4)
    candidates["Tail Risk Parity"] = tail_risk_parity_weights(r, max_weight)
    candidates["Robust Recommended"] = random_weight_search(r, "robust", max_weight, n_samples=n_samples, seed=seed + 5, tail_corr_penalty=0.25, concentration_penalty=0.50)
    rows = []
    for name, w in candidates.items():
        m = portfolio_metrics(r, w, name)
        m["Concentration HHI"] = float((w ** 2).sum())
        m["Max Weight"] = float(w.max())
        rows.append(m)
    metrics = pd.DataFrame(rows).set_index("Portfolio") if rows else pd.DataFrame()
    return candidates, metrics


def risk_contributions(returns: pd.DataFrame, weights: pd.Series | Dict[str, float]) -> pd.DataFrame:
    r = clean_returns(returns).fillna(0.0)
    cols = list(r.columns)
    w = normalize_weights(weights, cols)
    if r.empty or len(cols) == 0:
        return pd.DataFrame()
    cov = r.cov().values
    port_var = float(w.values.T @ cov @ w.values)
    if port_var <= 0:
        return pd.DataFrame()
    mcr = cov @ w.values
    vol_contrib = w.values * mcr / port_var
    p = r.dot(w)
    total_cvar = abs(cvar(p))
    cvar_contrib = []
    for c in cols:
        w2 = w.copy()
        w2[c] = 0.0
        w2 = normalize_weights(w2, cols)
        cvar_contrib.append(max(abs(cvar(p)) - abs(cvar(r.dot(w2))), 0.0))
    cvar_contrib = np.array(cvar_contrib)
    if cvar_contrib.sum() > 0:
        cvar_contrib = cvar_contrib / cvar_contrib.sum()
    out = pd.DataFrame({"Strategy": cols, "Weight": w.values, "Volatility Contribution": vol_contrib, "Approx cVaR Contribution": cvar_contrib})
    return out.sort_values("Volatility Contribution", ascending=False)


def synthetic_correlation_stress(returns: pd.DataFrame, weights: pd.Series | Dict[str, float], intensity: float = 0.75, vol_multiplier: float = 1.5, shock: float = -0.10) -> Dict[str, Any]:
    r = clean_returns(returns).fillna(0.0)
    cols = list(r.columns)
    w = normalize_weights(weights, cols)
    if r.empty:
        return {}
    base = portfolio_returns(r, w, "Base")
    common = r.mean(axis=1)
    stressed = (1.0 - intensity) * r + intensity * pd.DataFrame({c: common for c in cols}, index=r.index)
    stressed = stressed * vol_multiplier
    # Add a synthetic one-period common shock near the beginning for drawdown stress.
    if len(stressed) > 5 and shock != 0:
        shock_idx = stressed.index[min(5, len(stressed)-1)]
        stressed.loc[shock_idx, :] = stressed.loc[shock_idx, :] + shock
    sp = portfolio_returns(stressed, w, "Stressed")
    base_corr = r.corr()
    stress_corr = stressed.corr()
    res = {
        "base_series": base,
        "stress_series": sp,
        "base_corr": base_corr,
        "stress_corr": stress_corr,
        "base_metrics": metrics_for_series(base),
        "stress_metrics": metrics_for_series(sp),
        "avg_corr_base": float(base_corr.where(np.triu(np.ones(base_corr.shape), 1).astype(bool)).stack().mean()) if len(cols) > 1 else 1.0,
        "avg_corr_stress": float(stress_corr.where(np.triu(np.ones(stress_corr.shape), 1).astype(bool)).stack().mean()) if len(cols) > 1 else 1.0,
    }
    return res


def _bootstrap_indices(n: int, horizon: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    idxs = []
    while len(idxs) < horizon:
        start = int(rng.integers(0, max(n - block_len + 1, 1)))
        idxs.extend(range(start, min(start + block_len, n)))
    return np.array(idxs[:horizon]) % n


def monte_carlo_composite(
    returns: pd.DataFrame,
    weights: pd.Series | Dict[str, float],
    n_sims: int = 2000,
    horizon_years: int = 5,
    frequency: str = "Weekly",
    block_len: int = 8,
    seed: int = 123,
    return_haircut: float = 0.50,
    vol_multiplier: float = 1.25,
    corr_stress: float = 0.25,
    common_shock_prob: float = 0.02,
    common_shock_severity: float = -0.08,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = clean_returns(returns).fillna(0.0)
    if frequency == "Weekly":
        r = (1 + r).resample("W-FRI").prod() - 1
        periods_per_year = 52
    elif frequency == "Monthly":
        r = (1 + r).resample("ME").prod() - 1
        periods_per_year = 12
    else:
        periods_per_year = TRADING_DAYS
    r = r.dropna(how="all").fillna(0.0)
    cols = list(r.columns)
    w = normalize_weights(weights, cols)
    n = len(r)
    horizon = int(horizon_years * periods_per_year)
    rng = np.random.default_rng(seed)
    if n < 5 or horizon < 2:
        return pd.DataFrame(), pd.DataFrame()
    arr = r.values
    # Remove part of historical drift without reducing tails too much.
    drift = np.nanmean(arr, axis=0)
    arr = (arr - drift) * vol_multiplier + drift * return_haircut
    paths = np.empty((horizon + 1, n_sims), dtype=float)
    paths[0, :] = 1.0
    maxdds = []
    cagr_list = []
    final_vals = []
    time_underwater = []
    worst_period = []
    for s in range(n_sims):
        idx = _bootstrap_indices(n, horizon, max(1, int(block_len)), rng)
        sim = arr[idx, :].copy()
        if corr_stress > 0:
            common = sim.mean(axis=1, keepdims=True)
            sim = (1 - corr_stress) * sim + corr_stress * common
        if common_shock_prob > 0 and common_shock_severity != 0:
            mask = rng.random(horizon) < common_shock_prob
            sim[mask, :] += common_shock_severity
        pr = sim @ w.values
        wealth = np.concatenate([[1.0], np.cumprod(1 + pr)])
        paths[:, s] = wealth
        peak = np.maximum.accumulate(wealth)
        dd = wealth / peak - 1
        maxdds.append(float(dd.min()))
        final_vals.append(float(wealth[-1]))
        cagr_list.append(float(wealth[-1] ** (1 / horizon_years) - 1) if wealth[-1] > 0 else -1.0)
        time_underwater.append(float((dd < 0).mean()))
        worst_period.append(float(pr.min()))
    index = pd.RangeIndex(horizon + 1, name="Period")
    paths_df = pd.DataFrame(paths, index=index, columns=[f"sim_{i+1}" for i in range(n_sims)])
    summary = pd.DataFrame({
        "Final Wealth Multiple": final_vals,
        "CAGR": cagr_list,
        "Max Drawdown": maxdds,
        "Time Underwater %": time_underwater,
        "Worst Period Return": worst_period,
    })
    return paths_df, summary


def monte_carlo_statistics(summary: pd.DataFrame, dd_limit: float = -0.30, cagr_target: float = 0.15) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for col in summary.columns:
        s = summary[col].dropna()
        rows.append({
            "Metric": col,
            "Mean": s.mean(),
            "Median": s.median(),
            "1%": s.quantile(0.01),
            "5%": s.quantile(0.05),
            "25%": s.quantile(0.25),
            "75%": s.quantile(0.75),
            "95%": s.quantile(0.95),
            "99%": s.quantile(0.99),
        })
    extra = pd.DataFrame([
        {"Metric": f"P(Max DD < {dd_limit:.0%})", "Mean": float((summary["Max Drawdown"] < dd_limit).mean())},
        {"Metric": f"P(CAGR > {cagr_target:.0%})", "Mean": float((summary["CAGR"] > cagr_target).mean())},
        {"Metric": "P(Final wealth < 1x)", "Mean": float((summary["Final Wealth Multiple"] < 1).mean())},
    ])
    return pd.concat([pd.DataFrame(rows), extra], ignore_index=True)


def candidate_decision_table(metrics: pd.DataFrame, stress: Optional[pd.DataFrame] = None, mc: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    df = metrics.copy()
    # Normalize score components rank-wise to avoid hidden scale assumptions.
    def pct_rank(s, ascending=True):
        return s.rank(pct=True, ascending=ascending).fillna(0.5)
    score = (
        pct_rank(df["CAGR"], ascending=True) * 1.5
        + pct_rank(df["Sharpe"], ascending=True) * 1.0
        + pct_rank(df["Calmar"], ascending=True) * 1.0
        + pct_rank(df["Max Drawdown"], ascending=True) * 1.2  # less negative is better, ascending rank gives more negative lower
        + pct_rank(df["cVaR 95"], ascending=True) * 1.2
        - pct_rank(df["Concentration HHI"], ascending=True) * 0.5
    )
    df["Decision Score"] = score
    df["Classification"] = pd.cut(df["Decision Score"], bins=[-np.inf, score.quantile(.35), score.quantile(.75), np.inf], labels=["Research only", "Viable", "Recommended candidate"])
    return df.sort_values("Decision Score", ascending=False)
