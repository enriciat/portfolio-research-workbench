# Quantmage Backtest & Portfolio Research Workbench

A Streamlit app for running one or many Quantmage/Rainyboy strategy backtests, comparing the generated strategies, constructing candidate composite portfolios, stress-testing correlations, running Monte Carlo simulations, and downloading research-ready reports and CSV outputs.

## Key capabilities

- Upload one or many Quantmage JSON exports.
- Detect strategies inside each export and choose which ones to run.
- Run strategies in parallel with per-task status reporting.
- Analyze a previous output ZIP without rerunning backtests.
- Download individual strategy reports, equity CSVs, allocation CSVs, diagnostics, and a combined ZIP.
- Safe report preview with fallback Streamlit-native preview when standalone HTML reports render blank in an iframe.
- Portfolio Lab with candidate portfolios:
  - Equal weight
  - Inverse volatility
  - Inverse drawdown
  - Minimum variance
  - Maximum diversification
  - Max Sharpe
  - Max Sortino
  - Max Calmar
  - Min drawdown
  - Min cVaR
  - Tail-risk parity
  - Robust recommended
  - Manual weights
- Optional optimization/validation split for portfolio construction.
- Correlation & Synthetic Stress Lab:
  - Normal correlation matrix
  - Tail correlation matrix
  - Redundancy/similarity detector
  - Hierarchical clustering
  - Synthetic correlation-to-one stress
  - Volatility multiplier and common-shock stress
- Monte Carlo Lab for the active composite portfolio:
  - Daily/weekly/monthly bootstrap
  - Block bootstrap
  - Return haircut
  - Volatility multiplier
  - Correlation stress
  - Common shock probability/severity
  - Fan chart, CAGR distribution, max drawdown distribution, final wealth distribution, time-underwater distribution, worst simulated paths
- Benchmark/factor analysis:
  - Native benchmark equity columns
  - Optional uploaded factor CSV
  - Static beta/correlation
  - Rolling beta/correlation
  - Multi-factor regression with coefficients, t-stats, R², annual alpha, and residual volatility
- Exposure analysis:
  - Latest holdings
  - Composite latest exposure
  - Holdings overlap matrix
  - Optional holdings risk contribution from uploaded asset return/price CSV
- Reports & Downloads:
  - Portfolio research memo HTML
  - Active portfolio weights CSV
  - Composite returns CSV
  - Monte Carlo exports when run

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Put these files at the GitHub repository root:

```text
app.py
modules/
qm_native_engine.py
qm_to_rain.py
config/
reports/
requirements.txt
README.md
```

Then set the Streamlit main file path to:

```text
app.py
```

## Notes

- Standalone generated HTML reports can be very large and sometimes render blank inside Streamlit if their JavaScript expects a full browser page. Use the safe internal preview or download the report.
- Portfolio optimizers are decision-support tools, not guarantees. Use the validation-period mode and synthetic stress tests to reduce overfitting risk.
- Parallel workers are useful for local cached data. If many strategies need online API/Yahoo fallback, reduce worker count to avoid rate limits.

## New decision-engine additions

This version adds the decision-oriented portfolio tools from the Strategy Weighting project ideas:

- **Correlation Requirement Lab**: estimates the maximum useful correlation each strategy/pair can tolerate before it stops improving Sharpe or volatility, plus correlation safety-margin heatmaps and portfolio-level correlation breakpoints.
- **Efficient Frontier & Allocation Map**: generates random portfolios, marks an approximate efficient frontier, shows the active portfolio, and visualizes allocations along the frontier.
- **Robustness Lab**: reruns portfolio objectives by calendar year to show allocation stability and whether optimizers keep selecting the same strategies or chase noise.
- **Portfolio Decision page**: combines historical metrics, transparent decision-score components, correlation safety, and optional Monte Carlo candidate comparison to shortlist the portfolios that are most likely worth running.
- **Research workbook export**: creates an XLSX workbook containing decision tables, candidate metrics, correlation requirements, stress breakpoints, frontier data, robustness outputs, and Monte Carlo comparison outputs.

The app is still a decision-support tool, not a guarantee. Its strongest use is comparing candidate portfolios under explicit assumptions and showing where those assumptions break.
