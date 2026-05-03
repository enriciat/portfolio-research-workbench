from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from qm_native_engine import collect_strategies, slugify
from modules import analytics_plus as ap
from modules import portfolio_lab as pl
from modules import decision_lab as dl

APP_ROOT = Path(__file__).resolve().parent
ENGINE_PATH = APP_ROOT / "qm_native_engine.py"
CONFIG_DIR = APP_ROOT / "config"
REPORT_TEMPLATE = APP_ROOT / "reports" / "Standard_Short_logic_5BjE8lUAueEOt9GM1KGn_vs_QQQ.html"

st.set_page_config(
    page_title="Quantmage Backtest & Analytics Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class TaskResult:
    task_id: int
    display_name: str
    input_file_name: str
    status: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed: float = 0.0
    error: str = ""
    output_count: int = 0
    output_bytes: int = 0


@dataclass
class BacktestRun:
    name: str
    timestamp: float
    stdout: str
    stderr: str
    returncode: int
    output_dir: str
    zip_path: str
    output_files: List[Dict[str, Any]]
    index: Dict[str, Any]
    input_file_name: str
    task_count: int
    parallel_strategy_level: bool
    task_results: List[TaskResult] = field(default_factory=list)
    wrapper_warnings: List[str] = field(default_factory=list)


@dataclass
class BacktestTask:
    task_id: int
    display_name: str
    input_file_name: str
    payload_bytes: bytes
    run_key: str = ""
    source: str = ""
    path: str = ""
    depth: int = 0


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def safe_read(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def parse_uploaded_json(data: bytes) -> Tuple[Optional[Any], Optional[str]]:
    for enc in ["utf-8", "utf-8-sig"]:
        try:
            return json.loads(data.decode(enc)), None
        except Exception as exc:
            last = exc
    return None, f"Invalid JSON: {last}"


def strategy_depth(path: str) -> int:
    return max(path.count("/") + path.count(".") + path.count("[") // 2, 0)


def extract_strategy_preview(payload: Any, source_name: str = "") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    try:
        strategies = collect_strategies(payload)
    except Exception:
        strategies = []
    for i, (path, qm) in enumerate(strategies, start=1):
        run_key = f"{source_name}::{path}"
        rows.append({
            "Run": True,
            "#": i,
            "source": source_name,
            "name": str(qm.get("name") or f"strategy-{i}"),
            "path": path,
            "depth": strategy_depth(str(path)),
            "run_key": run_key,
            "benchmark": str(qm.get("benchmark_ticker") or "").upper(),
            "trading_type": str(qm.get("trading_type") or ""),
            "rebalance": str(qm.get("rebalance") or qm.get("rebalance_frequency") or ""),
        })
    return pd.DataFrame(rows)


def make_tasks(uploaded_payloads: List[Tuple[str, bytes, Any]], parallel_strategy_level: bool, selected_keys: Optional[List[str]] = None) -> List[BacktestTask]:
    selected = set(selected_keys or [])
    tasks: List[BacktestTask] = []
    next_id = 1
    for file_name, raw_bytes, payload in uploaded_payloads:
        if parallel_strategy_level:
            strategies = collect_strategies(payload)
            if strategies:
                for path, qm in strategies:
                    key = f"{file_name}::{path}"
                    if selected and key not in selected:
                        continue
                    qm_payload = json.loads(json.dumps(qm))  # detach from nested export
                    name = str(qm_payload.get("name") or f"strategy-{next_id}")
                    task_name = f"{Path(file_name).stem}__{slugify(name)}"
                    task_bytes = json.dumps(qm_payload, indent=2).encode("utf-8")
                    tasks.append(BacktestTask(next_id, name, f"{task_name}.json", task_bytes, key, file_name, str(path), strategy_depth(str(path))))
                    next_id += 1
                continue
        tasks.append(BacktestTask(next_id, Path(file_name).stem, file_name, raw_bytes, f"{file_name}::__whole_file__", file_name, "__whole_file__", 0))
        next_id += 1
    return tasks


def collect_output_files(out_dir: Path) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    if not out_dir.exists():
        return files
    for p in sorted([x for x in out_dir.rglob("*") if x.is_file()]):
        rel = p.relative_to(out_dir).as_posix()
        files.append({"path": rel, "suffix": p.suffix.lower(), "size": p.stat().st_size, "full_path": str(p)})
    return files


def zip_output_to_file(out_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted([x for x in out_dir.rglob("*") if x.is_file()]):
            if p == zip_path:
                continue
            zf.write(p, p.relative_to(out_dir).as_posix())


def build_index_from_outputs(out_dir: Path) -> Dict[str, Any]:
    stems = set()
    for p in out_dir.glob("*.report.html"):
        stems.add(p.name.replace(".report.html", ""))
    for p in out_dir.glob("*.equity.csv"):
        stems.add(p.name.replace(".equity.csv", ""))
    for p in out_dir.glob("*.allocation.csv"):
        stems.add(p.name.replace(".allocation.csv", ""))
    items: List[Dict[str, Any]] = []
    for stem in sorted(stems):
        item = {
            "name": stem.replace("-", " ").replace("_", " ").strip() or stem,
            "report_html": f"{stem}.report.html" if (out_dir / f"{stem}.report.html").exists() else None,
            "allocation_csv": f"{stem}.allocation.csv" if (out_dir / f"{stem}.allocation.csv").exists() else None,
            "equity_csv": f"{stem}.equity.csv" if (out_dir / f"{stem}.equity.csv").exists() else None,
            "symbols_csv": f"{stem}.symbols.csv" if (out_dir / f"{stem}.symbols.csv").exists() else None,
            "limiting_symbols_csv": f"{stem}.limiting_symbols.csv" if (out_dir / f"{stem}.limiting_symbols.csv").exists() else None,
            "start_diagnostics_txt": f"{stem}.start_diagnostics.txt" if (out_dir / f"{stem}.start_diagnostics.txt").exists() else None,
            "warnings_txt": f"{stem}.warnings.txt" if (out_dir / f"{stem}.warnings.txt").exists() else None,
            "warning_count": 0,
        }
        warn = item["warnings_txt"]
        if warn and (out_dir / warn).exists():
            item["warning_count"] = len([ln for ln in (out_dir / warn).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()])
        items.append(item)
    return {"count": len(items), "items": items}


def unique_destination(out_dir: Path, file_name: str) -> Path:
    dest = out_dir / file_name
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    for compound in [".report.html", ".equity.csv", ".allocation.csv", ".symbols.csv", ".limiting_symbols.csv", ".warnings.txt", ".start_diagnostics.txt"]:
        if file_name.endswith(compound):
            stem = file_name[: -len(compound)]
            suffix = compound
            break
    i = 2
    while True:
        candidate = out_dir / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def run_engine_task(
    task: BacktestTask,
    task_dir: Path,
    benchmark: str,
    chart_benchmark: str,
    strict: bool,
    use_testfolio_api: bool,
    use_yahoo_fallback: bool,
    allow_non_letf_proxy: bool,
    timeout_sec: int,
    retries: int,
    cookie: str,
    token: str,
) -> Tuple[BacktestTask, int, str, str, float, str]:
    task_dir.mkdir(parents=True, exist_ok=True)
    input_path = task_dir / task.input_file_name
    out_dir = task_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(task.payload_bytes)

    cmd = [
        sys.executable,
        str(ENGINE_PATH),
        "-i",
        str(input_path),
        "-o",
        str(out_dir),
        "--config",
        str(CONFIG_DIR),
        "--testfolio-timeout",
        str(float(timeout_sec)),
        "--testfolio-retries",
        str(int(retries)),
    ]
    if benchmark.strip():
        cmd += ["--benchmark", benchmark.strip().upper()]
    if chart_benchmark.strip():
        cmd += ["--chart-benchmark", chart_benchmark.strip().upper()]
    if strict:
        cmd.append("--strict")
    if not use_testfolio_api:
        cmd.append("--no-testfolio-api")
    if not use_yahoo_fallback:
        cmd.append("--no-yahoo-fallback")
    if allow_non_letf_proxy:
        cmd.append("--allow-non-letf-proxy")

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    if cookie:
        env["TESTFOLIO_COOKIE"] = cookie
    if token:
        env["TESTFOLIO_TOKEN"] = token

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            text=True,
            capture_output=True,
            timeout=max(30, timeout_sec * max(1, retries + 1) * 8),
            env=env,
        )
        return task, proc.returncode, proc.stdout, proc.stderr, time.time() - start, ""
    except subprocess.TimeoutExpired as exc:
        return task, 124, exc.stdout or "", exc.stderr or "", time.time() - start, f"Timed out: {exc}"
    except Exception as exc:
        return task, 1, "", "", time.time() - start, f"Worker exception: {exc}"


def analyze_existing_zip(uploaded_zip) -> BacktestRun:
    run_dir = Path(tempfile.mkdtemp(prefix="qm_existing_outputs_"))
    merged_out = run_dir / "merged_output"
    merged_out.mkdir(parents=True, exist_ok=True)
    raw = uploaded_zip.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        for member in zf.infolist():
            target = (merged_out / member.filename).resolve()
            if not str(target).startswith(str(merged_out.resolve())):
                raise ValueError(f"Unsafe ZIP path blocked: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    index = build_index_from_outputs(merged_out)
    if not (merged_out / "index.json").exists():
        (merged_out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    zip_path = run_dir / "outputs.zip"
    zip_output_to_file(merged_out, zip_path)
    files = collect_output_files(merged_out)
    return BacktestRun(
        name="existing_outputs",
        timestamp=time.time(),
        stdout="[Streamlit wrapper] Loaded existing output ZIP; no backtest run executed.",
        stderr="",
        returncode=0,
        output_dir=str(merged_out),
        zip_path=str(zip_path),
        output_files=files,
        index=index,
        input_file_name=str(uploaded_zip.name),
        task_count=0,
        parallel_strategy_level=False,
        task_results=[],
        wrapper_warnings=[],
    )


def run_backtest_batch(
    uploaded_payloads: List[Tuple[str, bytes, Any]],
    benchmark: str,
    chart_benchmark: str,
    strict: bool,
    use_testfolio_api: bool,
    use_yahoo_fallback: bool,
    allow_non_letf_proxy: bool,
    timeout_sec: int,
    retries: int,
    cookie: str,
    token: str,
    parallel_strategy_level: bool,
    max_workers: int,
    selected_strategy_keys: Optional[List[str]] = None,
    progress_placeholder=None,
) -> BacktestRun:
    tasks = make_tasks(uploaded_payloads, parallel_strategy_level=parallel_strategy_level, selected_keys=selected_strategy_keys)
    if not tasks:
        raise ValueError("No backtest tasks could be created from the selected uploads/strategies.")

    run_dir = Path(tempfile.mkdtemp(prefix="qm_backtest_batch_"))
    merged_out = run_dir / "merged_output"
    merged_out.mkdir(parents=True, exist_ok=True)
    task_root = run_dir / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)

    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    task_results: List[TaskResult] = []
    wrapper_warnings: List[str] = []
    completed = 0
    start_all = time.time()

    def process_tuple(result_tuple: Tuple[BacktestTask, int, str, str, float, str]) -> None:
        nonlocal completed
        task, code, out, err, elapsed, error = result_tuple
        task_out = task_root / f"task_{task.task_id}" / "output"
        output_count = 0
        output_bytes = 0
        if task_out.exists():
            for p in sorted([x for x in task_out.rglob("*") if x.is_file()]):
                if p.name == "index.json":
                    continue
                dest = unique_destination(merged_out, p.name)
                shutil.copy2(p, dest)
                output_count += 1
                output_bytes += p.stat().st_size
        status = "success" if code == 0 and not error else ("failed" if code != 0 else "warning")
        header = f"\n===== Task {task.task_id}: {task.display_name} ({elapsed:.1f}s, exit {code}, {status}) =====\n"
        stdout_parts.append(header + (out or ""))
        if err.strip() or error:
            stderr_parts.append(header + (error + "\n" if error else "") + (err or ""))
        task_results.append(TaskResult(task.task_id, task.display_name, task.input_file_name, status, code, out or "", err or "", elapsed, error or "", output_count, output_bytes))
        completed += 1
        if progress_placeholder is not None:
            progress_placeholder.progress(completed / len(tasks), text=f"Completed {completed}/{len(tasks)} task(s)")

    if parallel_strategy_level and len(tasks) > 1 and max_workers > 1:
        workers = min(max_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for task in tasks:
                futures.append(ex.submit(run_engine_task, task, task_root / f"task_{task.task_id}", benchmark, chart_benchmark, strict, use_testfolio_api, use_yahoo_fallback, allow_non_letf_proxy, timeout_sec, retries, cookie, token))
            for fut in as_completed(futures):
                try:
                    process_tuple(fut.result())
                except Exception as exc:
                    # Should be rare because run_engine_task catches internally, but keep the batch alive.
                    wrapper_warnings.append(f"A worker future failed outside the task wrapper: {exc}")
                    completed += 1
    else:
        for task in tasks:
            process_tuple(run_engine_task(task, task_root / f"task_{task.task_id}", benchmark, chart_benchmark, strict, use_testfolio_api, use_yahoo_fallback, allow_non_letf_proxy, timeout_sec, retries, cookie, token))

    index = build_index_from_outputs(merged_out)
    files_before = collect_output_files(merged_out)
    returns_df, _bench_df, _map = ap.load_returns_from_outputs(files_before, index)
    allocations = ap.load_latest_allocations(files_before, index)
    if not returns_df.empty:
        try:
            addendum = ap.generate_analytics_addendum_html(returns_df, allocations)
            (merged_out / "_multi_strategy_analytics_addendum.html").write_text(addendum, encoding="utf-8")
            ap.perf_summary(returns_df).to_csv(merged_out / "_analytics_performance_summary.csv")
            returns_df.to_csv(merged_out / "_strategy_returns.csv")
            returns_df.corr().to_csv(merged_out / "_correlation_matrix.csv")
            alloc_table = ap.latest_allocation_table(allocations)
            if not alloc_table.empty:
                alloc_table.to_csv(merged_out / "_latest_allocations.csv", index=False)
        except Exception as exc:
            wrapper_warnings.append(f"Analytics export generation failed: {exc}")
    else:
        wrapper_warnings.append("No strategy returns could be loaded from equity CSV outputs.")

    (merged_out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    files = collect_output_files(merged_out)
    zip_path = run_dir / "quantmage_backtest_outputs.zip"
    zip_output_to_file(merged_out, zip_path)
    elapsed_all = time.time() - start_all
    stdout = "".join(stdout_parts) + f"\n[Streamlit wrapper] Runtime: {elapsed_all:.1f}s; tasks: {len(tasks)}\n"
    stderr = "".join(stderr_parts)
    overall = 0 if task_results and all(t.returncode == 0 for t in task_results) and not wrapper_warnings else 1
    return BacktestRun("batch_backtest", time.time(), stdout, stderr, overall, str(merged_out), str(zip_path), files, index, "backtest_batch", len(tasks), parallel_strategy_level, task_results, wrapper_warnings)


def file_by_path(run: BacktestRun, path: str) -> Optional[Dict[str, Any]]:
    normalized = path.replace("\\", "/")
    for f in run.output_files:
        if f["path"] == normalized:
            return f
    return None


def render_task_status(run: BacktestRun) -> None:
    if not run.task_results and not run.wrapper_warnings:
        return
    st.subheader("Run diagnostics")
    if run.task_results:
        df = pd.DataFrame([t.__dict__ for t in run.task_results])
        show = df[["task_id", "display_name", "status", "returncode", "elapsed", "output_count", "output_bytes", "error"]].copy()
        show["output_bytes"] = show["output_bytes"].map(human_size)
        st.dataframe(show, use_container_width=True, hide_index=True)
        status_counts = show["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(px.bar(status_counts, x="Status", y="Count", title="Task status counts", text="Count"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(df, x="elapsed", nbins=20, title="Task runtime distribution"), use_container_width=True)
    if run.wrapper_warnings:
        st.warning("Wrapper warnings:\n" + "\n".join(f"- {w}" for w in run.wrapper_warnings))


def render_index(run: BacktestRun) -> None:
    items = run.index.get("items", []) if isinstance(run.index, dict) else []
    returns_df, _bench_df, _map = ap.load_returns_from_outputs(run.output_files, run.index)
    summary = ap.perf_summary(returns_df) if not returns_df.empty else pd.DataFrame()
    if not items:
        st.info("No index entries were produced. Check the logs below for details.")
        return
    rows: List[Dict[str, Any]] = []
    for item in items:
        name = item.get("name", "")
        row = {"Strategy": name, "Warnings": item.get("warning_count", 0), "Report": item.get("report_html", "")}
        s = None
        if name in summary.index:
            s = summary.loc[name]
        else:
            for idx in summary.index:
                if slugify(str(idx)) == slugify(str(name)):
                    s = summary.loc[idx]
                    break
        if s is not None:
            row.update({
                "Start": str(s.get("Start", "")),
                "End": str(s.get("End", "")),
                "CAGR": float(s.get("CAGR", np.nan)),
                "Max DD": float(s.get("Max Drawdown", np.nan)),
                "Sharpe": float(s.get("Sharpe", np.nan)),
                "Sortino": float(s.get("Sortino", np.nan)),
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    fmt = {"CAGR": "{:.2%}", "Max DD": "{:.2%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}"}
    st.dataframe(df.style.format({k: v for k, v in fmt.items() if k in df.columns}), use_container_width=True, hide_index=True)
    if "Warnings" in df.columns and df["Warnings"].sum() > 0:
        st.plotly_chart(px.bar(df.sort_values("Warnings", ascending=False), x="Strategy", y="Warnings", title="Warnings by strategy"), use_container_width=True)


def render_downloads(run: BacktestRun) -> None:
    if run.zip_path and Path(run.zip_path).exists():
        st.download_button("Download all outputs as ZIP", safe_read(run.zip_path), file_name="quantmage_backtest_outputs.zip", mime="application/zip", use_container_width=True)
    if not run.output_files:
        return
    st.subheader("Generated files")
    by_type = {
        "Reports": [f for f in run.output_files if f["path"].lower().endswith(".html")],
        "Equity CSVs": [f for f in run.output_files if f["path"].endswith(".equity.csv")],
        "Allocation CSVs": [f for f in run.output_files if f["path"].endswith(".allocation.csv")],
        "Analytics exports": [f for f in run.output_files if f["path"].startswith("_analytics") or f["path"] in {"_strategy_returns.csv", "_correlation_matrix.csv", "_latest_allocations.csv"}],
        "Diagnostics": [f for f in run.output_files if f["path"].endswith(".txt") or f["path"].endswith(".json") or f["path"].endswith(".symbols.csv") or f["path"].endswith(".limiting_symbols.csv")],
    }
    file_counts = pd.DataFrame([{"Type": k, "Count": len(v), "Size": sum(f["size"] for f in v)} for k, v in by_type.items() if v])
    if not file_counts.empty:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(file_counts, x="Type", y="Count", title="Output files by type", text="Count"), use_container_width=True)
        c2.plotly_chart(px.pie(file_counts, names="Type", values="Size", title="Output size by type"), use_container_width=True)
    for label, files in by_type.items():
        if not files:
            continue
        with st.expander(f"{label} ({len(files)})", expanded=(label in {"Reports", "Analytics exports"})):
            for f in files:
                col1, col2, col3 = st.columns([5, 1.2, 1.8])
                col1.code(f["path"], language=None)
                col2.write(human_size(f["size"]))
                mime = "text/html" if f["path"].endswith(".html") else "text/plain"
                if f["path"].endswith(".csv"):
                    mime = "text/csv"
                elif f["path"].endswith(".json"):
                    mime = "application/json"
                col3.download_button("Download", safe_read(f["full_path"]), file_name=Path(f["path"]).name, mime=mime, key=f"download_{run.timestamp}_{f['path']}")



def render_report_preview(run: BacktestRun) -> None:
    reports = [f for f in run.output_files if f["path"].lower().endswith(".html")]
    if not reports:
        st.info("No HTML reports available for preview.")
        return
    selected = st.selectbox("Preview report", [f["path"] for f in reports])
    f = file_by_path(run, selected)
    if not f:
        return
    file_size = Path(f["full_path"]).stat().st_size if Path(f["full_path"]).exists() else 0
    html_text = safe_read(f["full_path"]).decode("utf-8", errors="ignore")
    health = pd.DataFrame([
        {"Check": "File exists", "Value": bool(Path(f["full_path"]).exists())},
        {"Check": "File size", "Value": human_size(file_size)},
        {"Check": "Contains <script>", "Value": "<script" in html_text.lower()},
        {"Check": "Mentions Plotly", "Value": "plotly" in html_text.lower()},
        {"Check": "Fallback preview available", "Value": bool(run.index.get("items"))},
    ])
    st.dataframe(health, use_container_width=True, hide_index=True)
    mode = st.radio("Preview mode", ["Safe internal preview", "Native HTML iframe", "HTML source snippet"], horizontal=True)
    if mode == "Native HTML iframe":
        st.warning("Some standalone reports render blank inside Streamlit if their JavaScript expects a full browser page. Use the safe preview or download the report if this pane is blank.")
        height = st.slider("Preview height", 500, 1800, 1000, 100)
        components.html(html_text, height=height, scrolling=True)
        st.download_button("Download this HTML report", safe_read(f["full_path"]), file_name=Path(selected).name, mime="text/html", use_container_width=True)
        return
    if mode == "HTML source snippet":
        st.download_button("Download this HTML report", safe_read(f["full_path"]), file_name=Path(selected).name, mime="text/html", use_container_width=True)
        st.code(html_text[:30000], language="html")
        return

    st.subheader("Safe internal preview")
    item = None
    stem = Path(selected).name
    for suffix in [".report.html", "_report.html", ".html"]:
        if stem.endswith(suffix):
            stem_guess = stem[:-len(suffix)]
            break
    else:
        stem_guess = Path(selected).stem
    for it in run.index.get("items", []):
        eq = str(it.get("equity_csv") or "")
        rep = str(it.get("report_html") or "")
        nm = str(it.get("name") or "")
        if rep == selected or stem_guess in {eq.replace(".equity.csv", ""), nm, nm.replace(" ", "_")}:
            item = it
            break
    if selected.endswith("_addendum.html"):
        components.html(html_text, height=900, scrolling=True)
        return
    if item is None:
        # Fallback: use the first item with an equity CSV so previews never go blank when generic HTML report names differ.
        candidates = [it for it in run.index.get("items", []) if it.get("equity_csv")]
        item = candidates[0] if candidates else None
    if item is None:
        st.warning("Could not connect this report to equity/allocation CSV. The standalone download is still available.")
        st.download_button("Download this HTML report", safe_read(f["full_path"]), file_name=Path(selected).name, mime="text/html", use_container_width=True)
        return
    eq_path = item.get("equity_csv")
    alloc_path = item.get("allocation_csv")
    warn_path = item.get("warnings_txt")
    diag_path = item.get("start_diagnostics_txt")
    cols = st.columns(4)
    cols[0].metric("Strategy", str(item.get("name", ""))[:30])
    cols[1].metric("Warnings", item.get("warning_count", 0))
    cols[2].metric("Equity CSV", "yes" if eq_path else "no")
    cols[3].metric("Allocation CSV", "yes" if alloc_path else "no")
    if eq_path:
        eq_file = file_by_path(run, eq_path)
        if eq_file:
            try:
                eq = pd.read_csv(eq_file["full_path"])
                eq["Date"] = pd.to_datetime(eq["Date"], errors="coerce")
                eq = eq.dropna(subset=["Date"]).set_index("Date")
                plot_cols = [c for c in ["Strategy Equity", "Benchmark Equity"] if c in eq.columns]
                if plot_cols:
                    st.plotly_chart(px.line(eq[plot_cols], title="Equity preview", log_y=True), use_container_width=True)
                if "Strategy Equity" in eq.columns:
                    r = pd.to_numeric(eq["Strategy Equity"], errors="coerce").pct_change().dropna()
                    summary = ap.perf_summary(pd.DataFrame({str(item.get("name", "Strategy")): r}))
                    if not summary.empty:
                        st.dataframe(summary.style.format({c: "{:.4f}" for c in summary.select_dtypes("number").columns}), use_container_width=True)
                    st.plotly_chart(px.line(ap.drawdowns(pd.DataFrame({"Drawdown": r})), title="Drawdown preview"), use_container_width=True)
            except Exception as exc:
                st.error(f"Safe equity preview failed: {exc}")
    if alloc_path:
        af = file_by_path(run, alloc_path)
        if af:
            try:
                adf = pd.read_csv(af["full_path"])
                st.subheader("Allocation file preview")
                st.dataframe(adf.tail(25), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.warning(f"Could not preview allocation CSV: {exc}")
    if warn_path:
        wf = file_by_path(run, warn_path)
        if wf:
            txt = safe_read(wf["full_path"]).decode("utf-8", errors="ignore")
            with st.expander("Warnings", expanded=bool(txt.strip())):
                st.text(txt[:20000] or "No warnings")
    if diag_path:
        df = file_by_path(run, diag_path)
        if df:
            with st.expander("Start diagnostics", expanded=False):
                st.text(safe_read(df["full_path"]).decode("utf-8", errors="ignore")[:20000])
    st.download_button("Download full standalone HTML report", safe_read(f["full_path"]), file_name=Path(selected).name, mime="text/html", use_container_width=True)

def build_composite_ui(returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, float]]:
    if returns_df.shape[1] < 2:
        return returns_df, None, {}
    with st.expander("Composite portfolio builder", expanded=True):
        method = st.selectbox("Composite weighting", ["Equal weight", "Inverse volatility", "Manual weights"], index=0)
        cols = list(returns_df.columns)
        if method == "Equal weight":
            weights = {c: 1.0 / len(cols) for c in cols}
        elif method == "Inverse volatility":
            vols = returns_df.std().replace(0, np.nan)
            inv = 1 / vols
            inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0)
            if inv.sum() == 0:
                weights = {c: 1.0 / len(cols) for c in cols}
            else:
                weights = (inv / inv.sum()).to_dict()
        else:
            st.caption("Manual weights are normalized automatically.")
            weights = {c: st.number_input(f"Weight: {c}", 0.0, 100.0, 100.0 / len(cols), 5.0, key=f"cw_{c}") / 100 for c in cols}
        weight_df = pd.DataFrame({"Strategy": list(weights.keys()), "Weight": list(weights.values())})
        c1, c2 = st.columns([1, 1])
        c1.dataframe(weight_df.style.format({"Weight": "{:.2%}"}), use_container_width=True, hide_index=True)
        c2.plotly_chart(px.pie(weight_df, names="Strategy", values="Weight", title="Composite weights"), use_container_width=True)
        comp = ap.composite_returns(returns_df, weights)
        out = returns_df.copy()
        out[comp.name] = comp
        return out, comp, weights


def render_visual_metric_companions(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    df = summary.reset_index()
    c1, c2 = st.columns(2)
    with c1:
        if "CAGR" in df.columns and "Max Drawdown" in df.columns:
            st.plotly_chart(px.scatter(df, x="Max Drawdown", y="CAGR", size="Volatility", hover_name="Strategy", title="Risk/return map"), use_container_width=True)
    with c2:
        if "Sharpe" in df.columns:
            st.plotly_chart(px.bar(df.sort_values("Sharpe", ascending=False), x="Strategy", y="Sharpe", title="Sharpe ranking"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        if "Max Drawdown" in df.columns:
            st.plotly_chart(px.bar(df.sort_values("Max Drawdown"), x="Strategy", y="Max Drawdown", title="Max drawdown ranking"), use_container_width=True)
    with c4:
        if "cVaR 95" in df.columns:
            st.plotly_chart(px.bar(df.sort_values("cVaR 95"), x="Strategy", y="cVaR 95", title="Left-tail cVaR ranking"), use_container_width=True)



PERCENT_HINTS = (
    "return", "cagr", "vol", "drawdown", "dd", "weight", "var", "cvar",
    "win", "missing", "prob", "p(", "corr", "beta", "alpha", "exposure", "contribution",
    "time underwater", "final wealth"
)
RATIO_HINTS = ("sharpe", "sortino", "calmar", "score", "hhi", "multiple", "observations", "calendar years")


def readable_styler(df: pd.DataFrame, *, precision: int = 2):
    """Return a Styler with finance-aware formats and readable table defaults."""
    if df is None or df.empty:
        return df
    formats: Dict[str, str] = {}
    for col in df.columns:
        if not hasattr(df[col], "dtype") or df[col].dtype.kind not in "fcbi":
            continue
        low = str(col).lower()
        if any(h in low for h in PERCENT_HINTS) and not any(h in low for h in ("observations", "count")):
            formats[col] = "{:.2%}"
        elif any(h in low for h in RATIO_HINTS):
            formats[col] = "{:.2f}"
        else:
            formats[col] = "{:,.2f}"
    try:
        return df.style.format(formats, na_rep="—").set_table_styles([
            {"selector": "th", "props": [("font-weight", "700"), ("background-color", "#f4f6f8")]},
            {"selector": "td", "props": [("font-size", "12px")]},
        ])
    except Exception:
        return df


def display_readable_table(df: pd.DataFrame, *args, **kwargs) -> None:
    """Streamlit dataframe wrapper for consistently readable financial tables."""
    if df is None or df.empty:
        st.info("No table data available.")
        return
    st.dataframe(readable_styler(df), *args, **kwargs)


UNIVERSE_MODES = [
    "Full history",
    "Pre-training only",
    "Training only",
    "Post-training only",
    "Live only",
    "OOS + live, exclude training",
    "Custom date range",
]


def _bounded_date(dt, min_d, max_d):
    ts = pd.Timestamp(dt)
    return min(max(ts, pd.Timestamp(min_d)), pd.Timestamp(max_d)).date()


def _slice_by_universe(df: pd.DataFrame, mode: str, training_start, training_end, live_start, custom_start, custom_end) -> pd.DataFrame:
    """Slice returns according to the selected evidence universe while preserving columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    ts_train_start = pd.Timestamp(training_start)
    ts_train_end = pd.Timestamp(training_end)
    ts_live_start = pd.Timestamp(live_start)
    ts_custom_start = pd.Timestamp(custom_start)
    ts_custom_end = pd.Timestamp(custom_end)
    if mode == "Full history":
        mask = pd.Series(True, index=out.index)
    elif mode == "Pre-training only":
        mask = out.index < ts_train_start
    elif mode == "Training only":
        mask = (out.index >= ts_train_start) & (out.index <= ts_train_end)
    elif mode == "Post-training only":
        mask = out.index > ts_train_end
    elif mode == "Live only":
        mask = out.index >= ts_live_start
    elif mode == "OOS + live, exclude training":
        mask = (out.index < ts_train_start) | (out.index >= ts_live_start)
    elif mode == "Custom date range":
        mask = (out.index >= ts_custom_start) & (out.index <= ts_custom_end)
    else:
        mask = pd.Series(True, index=out.index)
    sliced = out.loc[mask].copy()
    return sliced.dropna(how="all")


def _universe_stats(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"Universe": name, "Start": "—", "End": "—", "Observations": 0, "Strategies": 0, "Missing %": np.nan}
    return {
        "Universe": name,
        "Start": df.index.min().date(),
        "End": df.index.max().date(),
        "Observations": int(len(df)),
        "Strategies": int(df.shape[1]),
        "Missing %": float(df.isna().mean().mean()),
    }


def render_data_universe_panel(base_returns: pd.DataFrame) -> Dict[str, Any]:
    """Central data-universe control panel for optimization, validation, simulation, and reporting."""
    if base_returns.empty:
        return {"analysis": base_returns, "optimization": base_returns, "validation": base_returns, "simulation": base_returns, "stress": base_returns, "labels": {}}
    min_date = base_returns.index.min().date()
    max_date = base_returns.index.max().date()
    span_days = max((pd.Timestamp(max_date) - pd.Timestamp(min_date)).days, 1)
    default_train_start = _bounded_date(pd.Timestamp(min_date) + pd.Timedelta(days=max(365, int(span_days * 0.65))), min_date, max_date)
    default_train_end = _bounded_date(pd.Timestamp(min_date) + pd.Timedelta(days=max(730, int(span_days * 0.85))), min_date, max_date)
    default_live_start = _bounded_date(pd.Timestamp(max_date) - pd.Timedelta(days=730), min_date, max_date)

    with st.expander("Data Universe Control Panel", expanded=True):
        st.caption("Use different evidence windows for optimization, validation, simulation, and reporting. This helps avoid choosing portfolio weights on the same period used to prove they work.")
        c1, c2, c3 = st.columns(3)
        training_start = c1.date_input("Training period starts", value=default_train_start, min_value=min_date, max_value=max_date, key="du_train_start")
        training_end = c2.date_input("Training period ends", value=default_train_end, min_value=min_date, max_value=max_date, key="du_train_end")
        live_start = c3.date_input("Live period starts", value=default_live_start, min_value=min_date, max_value=max_date, key="du_live_start")
        if pd.Timestamp(training_end) < pd.Timestamp(training_start):
            st.warning("Training end is before training start. The app will still slice, but this is probably not intended.")

        preset = st.selectbox(
            "Evidence preset",
            [
                "Best practice: optimize pre-training, validate live, simulate OOS+live, report full",
                "OOS + live evidence only; exclude training everywhere",
                "Full history everywhere",
                "Manual",
            ],
            index=0,
            key="du_preset",
        )
        custom_start = min_date
        custom_end = max_date
        show_custom_range = st.checkbox("Show custom range settings", value=False, key="du_show_custom_range")
        if show_custom_range:
            c4, c5 = st.columns(2)
            custom_start = c4.date_input("Custom range start", value=min_date, min_value=min_date, max_value=max_date, key="du_custom_start")
            custom_end = c5.date_input("Custom range end", value=max_date, min_value=min_date, max_value=max_date, key="du_custom_end")

        if preset == "Best practice: optimize pre-training, validate live, simulate OOS+live, report full":
            analysis_mode = "Full history"
            optimization_mode = "Pre-training only"
            validation_mode = "Live only"
            simulation_mode = "OOS + live, exclude training"
            stress_mode = "Full history"
        elif preset == "OOS + live evidence only; exclude training everywhere":
            analysis_mode = optimization_mode = validation_mode = simulation_mode = stress_mode = "OOS + live, exclude training"
        elif preset == "Full history everywhere":
            analysis_mode = optimization_mode = validation_mode = simulation_mode = stress_mode = "Full history"
        else:
            c6, c7, c8 = st.columns(3)
            analysis_mode = c6.selectbox("Reporting / performance universe", UNIVERSE_MODES, index=0, key="du_analysis")
            optimization_mode = c7.selectbox("Optimization universe", UNIVERSE_MODES, index=1 if len(base_returns) > 400 else 0, key="du_opt")
            validation_mode = c8.selectbox("Validation universe", UNIVERSE_MODES, index=4 if len(base_returns) > 400 else 0, key="du_val")
            c9, c10 = st.columns(2)
            simulation_mode = c9.selectbox("Monte Carlo / simulation universe", UNIVERSE_MODES, index=5 if len(base_returns) > 400 else 0, key="du_sim")
            stress_mode = c10.selectbox("Stress / correlation universe", UNIVERSE_MODES, index=0, key="du_stress")

        universes = {
            "analysis": _slice_by_universe(base_returns, analysis_mode, training_start, training_end, live_start, custom_start, custom_end),
            "optimization": _slice_by_universe(base_returns, optimization_mode, training_start, training_end, live_start, custom_start, custom_end),
            "validation": _slice_by_universe(base_returns, validation_mode, training_start, training_end, live_start, custom_start, custom_end),
            "simulation": _slice_by_universe(base_returns, simulation_mode, training_start, training_end, live_start, custom_start, custom_end),
            "stress": _slice_by_universe(base_returns, stress_mode, training_start, training_end, live_start, custom_start, custom_end),
        }
        modes = {"analysis": analysis_mode, "optimization": optimization_mode, "validation": validation_mode, "simulation": simulation_mode, "stress": stress_mode}
        stats = pd.DataFrame([_universe_stats(k.title(), v) | {"Mode": modes[k]} for k, v in universes.items()])
        display_readable_table(stats, use_container_width=True, hide_index=True)
        if any(v.empty for v in universes.values()):
            st.warning("At least one selected data universe is empty. Affected modules will fall back to the reporting universe where needed.")
        st.session_state["data_universe_modes"] = modes
    labels = {k: modes[k] for k in modes}
    return {"labels": labels, **universes}


def _safe_universe(df: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    return df if df is not None and not df.empty else fallback



def _candidate_weights_to_df(candidates: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for port, w in candidates.items():
        for strat, val in w.items():
            rows.append({"Portfolio": port, "Strategy": strat, "Weight": float(val)})
    return pd.DataFrame(rows)

def _apply_weight_caps(weights: pd.Series, caps: Dict[str, float]) -> pd.Series:
    """Clip weights to per-strategy caps and redistribute leftover to uncapped names."""
    w = weights.astype(float).copy().fillna(0.0).clip(lower=0.0)
    if w.sum() <= 0:
        return pl.normalize_weights(w, list(w.index))
    w = w / w.sum()
    caps_s = pd.Series({k: float(v) for k, v in caps.items()}, index=w.index).fillna(1.0).clip(lower=0.0, upper=1.0)
    for _ in range(100):
        over = w > caps_s + 1e-12
        if not over.any():
            break
        fixed = w[over].clip(upper=caps_s[over])
        remainder = 1.0 - fixed.sum()
        rest = w[~over]
        if remainder <= 0 or rest.empty or rest.sum() <= 0:
            w = pd.concat([fixed, pd.Series(0.0, index=rest.index)]).reindex(w.index).fillna(0.0)
            break
        rest = rest / rest.sum() * remainder
        w = pd.concat([fixed, rest]).reindex(w.index).fillna(0.0)
    return pl.normalize_weights(w, list(w.index))


def _get_active_weights(base_returns: pd.DataFrame) -> pd.Series:
    cols = list(base_returns.columns)
    saved = st.session_state.get("active_portfolio_weights")
    if isinstance(saved, dict):
        return pl.normalize_weights(saved, cols)
    return pl.normalize_weights({c: 1 / len(cols) for c in cols}, cols)


def _add_active_composite(base_returns: pd.DataFrame, name: str = "Composite Portfolio") -> Tuple[pd.DataFrame, pd.Series]:
    weights = _get_active_weights(base_returns)
    out = base_returns.copy()
    out[name] = pl.portfolio_returns(base_returns, weights, name)
    return out, weights


def render_portfolio_lab(base_returns: pd.DataFrame, opt_universe: Optional[pd.DataFrame] = None, validation_universe: Optional[pd.DataFrame] = None, universe_labels: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    st.header("Portfolio Lab")
    st.caption("Build candidate portfolios from the generated strategy return streams. This page is meant to answer: which strategy mix is best to actually run?")
    if base_returns.shape[1] < 2:
        w = pl.normalize_weights({base_returns.columns[0]: 1.0}, list(base_returns.columns))
        return _add_active_composite(base_returns)

    opt_returns = _safe_universe(opt_universe, base_returns)
    val_returns = _safe_universe(validation_universe, base_returns)
    universe_labels = universe_labels or {}
    with st.expander("Portfolio construction controls", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        max_weight = c1.slider("Max weight per strategy", 0.10, 1.00, 0.50, 0.05)
        n_samples = c2.slider("Random-search samples", 100, 3000, 600, 100)
        seed = c3.number_input("Optimizer seed", min_value=1, max_value=999999, value=42, step=1)
        include_manual = c4.checkbox("Create manual portfolio", value=True)
        st.info(
            f"Weights are optimized on **{universe_labels.get('optimization', 'selected optimization universe')}** "
            f"and evaluated on **{universe_labels.get('validation', 'selected validation universe')}**. "
            "This is stricter than optimizing and judging on the same period."
        )
        st.caption(f"Optimization observations: {len(opt_returns)} | Validation observations: {len(val_returns)} | Reporting observations: {len(base_returns)}")

    with st.expander("Per-strategy constraints", expanded=False):
        st.caption("Exclude strategies or cap their maximum weight. Caps are applied after every optimizer and before evaluation.")
        constraint_rows = []
        for col in base_returns.columns:
            cc1, cc2, cc3 = st.columns([3, 1, 1])
            cc1.write(col)
            excluded = cc2.checkbox("Exclude", value=False, key=f"exclude_{col}")
            cap = cc3.slider("Max", 0.0, 1.0, float(max_weight), 0.05, key=f"cap_{col}")
            constraint_rows.append({"Strategy": col, "Exclude": excluded, "Max Weight": cap})
        constraints_df = pd.DataFrame(constraint_rows)
    allowed_cols = constraints_df.loc[~constraints_df["Exclude"], "Strategy"].tolist()
    if not allowed_cols:
        st.error("All strategies are excluded. Include at least one strategy.")
        return _add_active_composite(base_returns)
    caps = dict(zip(constraints_df["Strategy"], constraints_df["Max Weight"]))
    opt_returns = opt_returns[allowed_cols]
    base_eval_returns = val_returns[allowed_cols].dropna(how="all")
    if base_eval_returns.empty:
        st.warning("Validation universe is empty after constraints; falling back to reporting universe for evaluation.")
        base_eval_returns = base_returns[allowed_cols].dropna(how="all")
    with st.spinner("Building candidate portfolios…"):
        candidates, metrics = pl.candidate_portfolios(opt_returns, max_weight=float(max_weight), n_samples=int(n_samples), seed=int(seed))
        candidates = {name: _apply_weight_caps(w.reindex(allowed_cols).fillna(0.0), caps) for name, w in candidates.items()}
    if include_manual:
        with st.expander("Manual portfolio weights", expanded=False):
            manual = {}
            default = 100.0 / len(allowed_cols)
            for c in allowed_cols:
                manual[c] = st.number_input(f"Manual weight: {c}", 0.0, 100.0, default, 5.0, key=f"manual_port_w_{c}") / 100.0
            candidates["Manual"] = pl.normalize_weights(manual, allowed_cols, max_weight=1.0)
            m = pl.portfolio_metrics(base_eval_returns, candidates["Manual"], "Manual")
            m["Concentration HHI"] = float((candidates["Manual"] ** 2).sum())
            m["Max Weight"] = float(candidates["Manual"].max())
            metrics = pd.concat([metrics, pd.DataFrame([m]).set_index("Portfolio")]) if not metrics.empty else pd.DataFrame([m]).set_index("Portfolio")

    # Evaluate candidate weights on the full selected universe and, if enabled, the validation period.
    eval_rows = []
    for nm, ww in candidates.items():
        mm = pl.portfolio_metrics(base_eval_returns, ww, nm)
        mm["Concentration HHI"] = float((ww ** 2).sum())
        mm["Max Weight"] = float(ww.max())
        eval_rows.append(mm)
    eval_metrics = pd.DataFrame(eval_rows).set_index("Portfolio") if eval_rows else metrics
    decision = pl.candidate_decision_table(eval_metrics)
    if decision.empty:
        st.warning("Could not build candidate portfolio metrics.")
        return _add_active_composite(base_returns)

    if len(base_eval_returns) > 30:
        val_rows = []
        for nm, ww in candidates.items():
            vm = pl.portfolio_metrics(base_eval_returns, ww, nm)
            vm["Concentration HHI"] = float((ww ** 2).sum())
            vm["Max Weight"] = float(ww.max())
            val_rows.append(vm)
        val_metrics = pd.DataFrame(val_rows).set_index("Portfolio") if val_rows else pd.DataFrame()
        if not val_metrics.empty:
            st.subheader("Validation-universe candidate performance")
            display_readable_table(val_metrics, use_container_width=True)
            vdf = val_metrics.reset_index()
            st.plotly_chart(px.scatter(vdf, x="Max Drawdown", y="CAGR", size="Volatility", hover_name="Portfolio", title="Validation-universe risk/return map"), use_container_width=True)

    st.subheader("Candidate portfolio decision table")
    num_cols = decision.select_dtypes("number").columns
    display_readable_table(decision, use_container_width=True)
    # Visual companions for the table.
    dfp = decision.reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(dfp, x="Max Drawdown", y="CAGR", size="Volatility", color="Classification", hover_name="Portfolio", title="Candidate risk/return map"), use_container_width=True)
    c2.plotly_chart(px.bar(dfp.sort_values("Decision Score", ascending=False), x="Portfolio", y="Decision Score", color="Classification", title="Decision-score ranking"), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(dfp.sort_values("cVaR 95"), x="Portfolio", y="cVaR 95", title="Candidate left-tail cVaR ranking"), use_container_width=True)
    c4.plotly_chart(px.bar(dfp.sort_values("Max Drawdown"), x="Portfolio", y="Max Drawdown", title="Candidate max-drawdown ranking"), use_container_width=True)

    selected_port = st.selectbox("Select active portfolio to analyze everywhere else", list(decision.index), index=0)
    st.session_state["active_portfolio_name"] = selected_port
    st.session_state["active_portfolio_weights"] = candidates[selected_port].to_dict()
    weights = candidates[selected_port]

    st.subheader(f"Selected weights: {selected_port}")
    wdf = pd.DataFrame({"Strategy": weights.index, "Weight": weights.values}).sort_values("Weight", ascending=False)
    c1, c2 = st.columns(2)
    c1.dataframe(readable_styler(wdf), use_container_width=True, hide_index=True)
    c2.plotly_chart(px.pie(wdf, names="Strategy", values="Weight", title="Selected portfolio weights", hole=0.35), use_container_width=True)
    st.plotly_chart(px.bar(wdf, x="Strategy", y="Weight", title="Selected portfolio weights"), use_container_width=True)

    rc = pl.risk_contributions(base_returns, weights)
    if not rc.empty:
        st.subheader("Risk contribution")
        display_readable_table(rc, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(rc, x="Strategy", y="Volatility Contribution", title="Volatility risk contribution"), use_container_width=True)
        c2.plotly_chart(px.bar(rc, x="Strategy", y="Approx cVaR Contribution", title="Approximate cVaR contribution"), use_container_width=True)

    cand_rets = pd.DataFrame({name: pl.portfolio_returns(base_eval_returns, w, name) for name, w in candidates.items()})
    st.subheader("Candidate equity and drawdown comparison")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line((1 + cand_rets.fillna(0)).cumprod(), title="Candidate equity curves", log_y=True), use_container_width=True)
    c2.plotly_chart(px.line(pl.drawdowns(cand_rets), title="Candidate drawdowns"), use_container_width=True)

    st.download_button("Download candidate metrics CSV", decision.to_csv().encode("utf-8"), file_name="portfolio_candidate_metrics.csv", mime="text/csv")
    st.session_state["portfolio_candidates"] = {k: v.to_dict() for k, v in candidates.items()}
    st.session_state["portfolio_candidate_metrics"] = decision
    st.session_state["portfolio_constraints"] = constraints_df.to_dict("records")
    st.download_button("Download candidate weights CSV", _candidate_weights_to_df(candidates).to_csv(index=False).encode("utf-8"), file_name="portfolio_candidate_weights.csv", mime="text/csv")
    return _add_active_composite(base_returns)


def render_correlation_stress_lab(base_returns: pd.DataFrame, returns_with_composite: pd.DataFrame, weights: pd.Series) -> None:
    st.header("Correlation & Synthetic Stress Lab")
    if base_returns.shape[1] < 2:
        st.warning("Need at least two strategies for correlation stress analysis.")
        return
    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)
    trigger = c2.selectbox("Tail trigger", returns_with_composite.columns, index=list(returns_with_composite.columns).index("Composite Portfolio") if "Composite Portfolio" in returns_with_composite.columns else 0)
    tail_threshold = c3.slider("Tail drawdown threshold", -0.60, -0.01, -0.05, 0.01)
    sim_threshold = c4.slider("Similarity warning threshold", 0.70, 0.99, 0.90, 0.01)
    corr = ap.correlation_matrix(returns_with_composite, method=method)
    tail_corr, n_tail = ap.tail_correlation(returns_with_composite, trigger=trigger, threshold=tail_threshold)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Normal correlation matrix"), use_container_width=True)
    c2.plotly_chart(px.imshow(tail_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=f"Tail correlation matrix ({n_tail} obs)"), use_container_width=True)
    sim = ap.similarity_pairs(base_returns, threshold=sim_threshold)
    if not sim.empty:
        st.warning("Potential strategy redundancy detected. These pairs may not provide true diversification.")
        st.dataframe(sim.style.format({"Correlation": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(sim, x="Strategy A", y="Correlation", color="Strategy B", title="Highly similar strategy pairs"), use_container_width=True)
    st.subheader("Hierarchical clustering")
    try:
        linkage = ap.cluster_linkage(base_returns)
        fig = ff.create_dendrogram(base_returns.corr(), labels=list(base_returns.columns), linkagefun=lambda _: linkage)
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not build dendrogram: {exc}")

    st.subheader("Synthetic correlation stress")
    s1, s2, s3 = st.columns(3)
    intensity = s1.slider("Correlation-to-one intensity", 0.0, 1.0, 0.75, 0.05)
    vol_mult = s2.slider("Volatility multiplier", 1.0, 4.0, 1.75, 0.25)
    common_shock = s3.slider("One-period common shock", -0.50, 0.0, -0.10, 0.01)
    stress = pl.synthetic_correlation_stress(base_returns, weights, intensity=float(intensity), vol_multiplier=float(vol_mult), shock=float(common_shock))
    if stress:
        normal = stress["base_metrics"]
        stressed = stress["stress_metrics"]
        stress_table = pd.DataFrame([normal, stressed], index=["Base composite", "Synthetic stress composite"])
        st.dataframe(stress_table.style.format({c: "{:.4f}" for c in stress_table.select_dtypes("number").columns}), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.imshow(stress["stress_corr"], text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Stressed correlation matrix"), use_container_width=True)
        comparison = pd.DataFrame({"Base": stress["base_series"], "Synthetic stress": stress["stress_series"]})
        c2.plotly_chart(px.line((1 + comparison).cumprod(), title="Base vs synthetic stress equity", log_y=True), use_container_width=True)
        mdf = stress_table.reset_index().rename(columns={"index": "Scenario"})
        c3, c4 = st.columns(2)
        c3.plotly_chart(px.bar(mdf, x="Scenario", y="Max Drawdown", title="Stress max drawdown impact"), use_container_width=True)
        c4.plotly_chart(px.bar(mdf, x="Scenario", y="cVaR 95", title="Stress cVaR impact"), use_container_width=True)

    st.subheader("Rolling pairwise correlation")
    left_col, right_col = st.columns(2)
    left = left_col.selectbox("First strategy", base_returns.columns, index=0)
    right = right_col.selectbox("Second strategy", base_returns.columns, index=min(1, len(base_returns.columns) - 1))
    window = st.slider("Rolling correlation window", 21, 504, 63)
    if left != right:
        st.plotly_chart(px.line(ap.rolling_corr(base_returns, left, right, window=window), title=f"Rolling correlation: {left} vs {right}"), use_container_width=True)


def render_monte_carlo_lab(base_returns: pd.DataFrame, weights: pd.Series) -> None:
    st.header("Monte Carlo Lab for Active Composite")
    st.caption("This simulation is portfolio-focused, not loan-focused. It stress-tests the selected composite using bootstrap blocks, alpha haircuts, volatility multipliers, correlation stress, and common shocks.")
    c1, c2, c3, c4 = st.columns(4)
    n_sims = c1.slider("Simulations", 250, 10000, 2000, 250)
    horizon = c2.slider("Horizon years", 1, 30, 10, 1)
    freq = c3.selectbox("Frequency", ["Weekly", "Monthly", "Daily"], index=0)
    block_len = c4.slider("Bootstrap block length", 1, 52, 8, 1)
    s1, s2, s3, s4 = st.columns(4)
    haircut = s1.slider("Return/alpha haircut", 0.0, 1.5, 0.50, 0.05)
    vol_mult = s2.slider("Volatility multiplier", 0.5, 4.0, 1.25, 0.05)
    corr_stress = s3.slider("Correlation stress", 0.0, 1.0, 0.25, 0.05)
    seed = s4.number_input("Random seed", 1, 999999, 123, 1)
    h1, h2, h3 = st.columns(3)
    shock_prob = h1.slider("Common shock probability per period", 0.0, 0.20, 0.02, 0.005)
    shock_sev = h2.slider("Common shock severity", -0.50, 0.0, -0.08, 0.01)
    dd_limit = h3.slider("Drawdown breach threshold", -0.80, -0.05, -0.30, 0.05)
    target_cagr = st.slider("Target CAGR for probability calculation", -0.20, 1.00, 0.15, 0.01)

    run_mc = st.button("Run Monte Carlo", type="primary", use_container_width=True)
    if run_mc or "last_mc_summary" in st.session_state:
        if run_mc:
            with st.spinner("Simulating composite portfolio paths…"):
                paths, summary = pl.monte_carlo_composite(base_returns, weights, n_sims=int(n_sims), horizon_years=int(horizon), frequency=freq, block_len=int(block_len), seed=int(seed), return_haircut=float(haircut), vol_multiplier=float(vol_mult), corr_stress=float(corr_stress), common_shock_prob=float(shock_prob), common_shock_severity=float(shock_sev))
            st.session_state["last_mc_paths"] = paths
            st.session_state["last_mc_summary"] = summary
        paths = st.session_state.get("last_mc_paths", pd.DataFrame())
        summary = st.session_state.get("last_mc_summary", pd.DataFrame())
        if paths.empty or summary.empty:
            st.warning("Monte Carlo did not produce usable paths.")
            return
        stats = pl.monte_carlo_statistics(summary, dd_limit=float(dd_limit), cagr_target=float(target_cagr))
        st.subheader("Simulation summary")
        st.dataframe(stats.style.format({c: "{:.4f}" for c in stats.select_dtypes("number").columns}), use_container_width=True, hide_index=True)
        # Fan chart percentiles
        pct = pd.DataFrame({
            "1%": paths.quantile(0.01, axis=1),
            "5%": paths.quantile(0.05, axis=1),
            "25%": paths.quantile(0.25, axis=1),
            "Median": paths.quantile(0.50, axis=1),
            "75%": paths.quantile(0.75, axis=1),
            "95%": paths.quantile(0.95, axis=1),
            "99%": paths.quantile(0.99, axis=1),
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pct.index, y=pct["95%"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=pct.index, y=pct["5%"], fill="tonexty", name="5%-95% band", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=pct.index, y=pct["75%"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=pct.index, y=pct["25%"], fill="tonexty", name="25%-75% band", line=dict(width=0)))
        fig.add_trace(go.Scatter(x=pct.index, y=pct["Median"], name="Median"))
        fig.update_layout(title="Monte Carlo composite equity fan chart", yaxis_title="Wealth multiple")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(summary, x="CAGR", nbins=60, title="Simulated CAGR distribution"), use_container_width=True)
        c2.plotly_chart(px.histogram(summary, x="Max Drawdown", nbins=60, title="Simulated max drawdown distribution"), use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(px.histogram(summary, x="Final Wealth Multiple", nbins=60, title="Final wealth multiple distribution"), use_container_width=True)
        c4.plotly_chart(px.histogram(summary, x="Time Underwater %", nbins=60, title="Time underwater distribution"), use_container_width=True)
        st.subheader("Worst simulated paths")
        worst_cols = summary.sort_values("Max Drawdown").head(10).index
        worst_paths = paths.iloc[:, list(worst_cols)]
        st.plotly_chart(px.line(worst_paths, title="Worst 10 simulated paths by max drawdown"), use_container_width=True)
        st.download_button("Download Monte Carlo summary CSV", summary.to_csv(index=False).encode("utf-8"), "monte_carlo_summary.csv", "text/csv")
        # Limit path export if huge; still useful.
        st.download_button("Download Monte Carlo paths CSV", paths.to_csv().encode("utf-8"), "monte_carlo_paths.csv", "text/csv")



def _candidate_dict_from_state(base_returns: pd.DataFrame) -> Dict[str, pd.Series]:
    saved = st.session_state.get("portfolio_candidates")
    if isinstance(saved, dict) and saved:
        return {name: pl.normalize_weights(pd.Series(w, dtype=float), list(base_returns.columns)) for name, w in saved.items()}
    candidates, _ = pl.candidate_portfolios(base_returns, max_weight=0.5, n_samples=600, seed=42)
    return candidates


def render_correlation_requirement_lab(base_returns: pd.DataFrame, weights: pd.Series) -> None:
    st.header("Correlation Requirement Lab")
    st.caption("This page turns correlation from a descriptive heatmap into a decision tool: how low must correlations stay for each strategy or strategy pair to be worth running?")
    if base_returns.shape[1] < 2:
        st.warning("Need at least two strategies.")
        return
    req = dl.marginal_correlation_requirements(base_returns, weights)
    st.subheader("Strategy-to-portfolio correlation requirements")
    st.dataframe(req.style.format({c: "{:.3f}" for c in req.select_dtypes("number").columns}), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(req, x="Corr to Portfolio", y="Max Corr for Sharpe Accretion", color="Decision", size="Weight", hover_name="Strategy", title="Current correlation vs max Sharpe-accretive correlation"), use_container_width=True)
    c2.plotly_chart(px.bar(req.sort_values("Sharpe Corr Safety"), x="Strategy", y="Sharpe Corr Safety", color="Decision", title="Sharpe correlation safety margin"), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(req.sort_values("Tail Corr to Portfolio", ascending=False), x="Strategy", y="Tail Corr to Portfolio", color="Decision", title="Tail correlation to active portfolio"), use_container_width=True)
    c4.plotly_chart(px.bar(req.sort_values("Vol Corr Safety"), x="Strategy", y="Vol Corr Safety", color="Decision", title="Volatility-reduction correlation safety"), use_container_width=True)

    corr, required, safety = dl.pairwise_required_correlation(base_returns)
    st.subheader("Pairwise required correlation matrices")
    a, b, c = st.columns(3)
    a.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Observed correlation"), use_container_width=True)
    b.plotly_chart(px.imshow(required, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Approx max useful correlation"), use_container_width=True)
    c.plotly_chart(px.imshow(safety, text_auto=".2f", color_continuous_scale="RdYlGn", title="Correlation safety margin"), use_container_width=True)
    danger = safety.where(~np.eye(len(safety), dtype=bool)).stack().reset_index()
    danger.columns = ["Strategy A", "Strategy B", "Safety Margin"]
    danger = danger.sort_values("Safety Margin").head(20)
    st.subheader("Most dangerous correlation pairs")
    display_readable_table(danger, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(danger, x="Strategy A", y="Safety Margin", color="Strategy B", title="Pairs closest to / above required correlation limits"), use_container_width=True)

    st.subheader("Portfolio-level correlation breakpoints")
    dd_threshold = st.slider("Max drawdown failure threshold", -0.80, -0.05, -0.30, 0.05, key="corr_req_dd")
    sharpe_threshold = st.slider("Minimum acceptable Sharpe", -1.0, 5.0, 1.0, 0.1, key="corr_req_sharpe")
    curve = dl.correlation_breakpoint_curve(base_returns, weights)
    if not curve.empty:
        display_readable_table(curve, use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["Avg Pairwise Corr"], y=curve["Max Drawdown"], name="Max Drawdown"))
        fig.add_trace(go.Scatter(x=curve["Avg Pairwise Corr"], y=curve["cVaR 95"], name="cVaR 95"))
        fig.add_hline(y=dd_threshold, line_dash="dash", annotation_text="DD threshold")
        fig.update_layout(title="How portfolio risk changes as correlations rise", xaxis_title="Average stressed pairwise correlation")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(curve, x="Avg Pairwise Corr", y="Sharpe", title="Sharpe as correlations rise"), use_container_width=True)
        c2.plotly_chart(px.line(curve, x="Avg Pairwise Corr", y="Volatility", title="Volatility as correlations rise"), use_container_width=True)
        fail_dd = curve[curve["Max Drawdown"] < dd_threshold]
        fail_sh = curve[curve["Sharpe"] < sharpe_threshold]
        msg = []
        if not fail_dd.empty:
            msg.append(f"Drawdown threshold first fails around avg corr {fail_dd.iloc[0]['Avg Pairwise Corr']:.2f}.")
        if not fail_sh.empty:
            msg.append(f"Sharpe threshold first fails around avg corr {fail_sh.iloc[0]['Avg Pairwise Corr']:.2f}.")
        if msg:
            st.warning(" ".join(msg))
        else:
            st.success("The selected portfolio stayed within the chosen thresholds across the tested correlation stress grid.")
    st.session_state["last_corr_requirements"] = req
    st.session_state["last_pairwise_corr"] = corr
    st.session_state["last_required_corr"] = required
    st.session_state["last_corr_safety"] = safety
    st.session_state["last_corr_breakpoints"] = curve


def render_efficient_frontier_lab(base_returns: pd.DataFrame, weights: pd.Series) -> None:
    st.header("Efficient Frontier & Allocation Map")
    st.caption("Random portfolios plus an approximate frontier. This shows where the active portfolio sits and how allocation changes as expected return rises.")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Random portfolios", 500, 15000, 4000, 500)
    max_w = c2.slider("Max weight in random portfolios", 0.10, 1.00, 0.60, 0.05, key="frontier_maxw")
    seed = c3.number_input("Frontier seed", 1, 999999, 7, 1)
    pts, wdf = dl.generate_random_portfolios(base_returns, n=int(n), max_weight=float(max_w), seed=int(seed))
    if pts.empty:
        st.warning("Could not generate frontier.")
        return
    active = pl.portfolio_metrics(base_returns, weights, "Active Portfolio")
    fig = px.scatter(pts, x="Volatility", y="CAGR", color="Max Drawdown", hover_name="Portfolio", title="Random portfolio cloud and efficient frontier", opacity=0.35)
    eff = pts[pts["Efficient"]].sort_values("Volatility")
    fig.add_trace(go.Scatter(x=eff["Volatility"], y=eff["CAGR"], mode="lines", name="Approx efficient frontier", line=dict(width=4)))
    fig.add_trace(go.Scatter(x=[active.get("Volatility")], y=[active.get("CAGR")], mode="markers", name="Active portfolio", marker=dict(size=16, symbol="star")))
    st.plotly_chart(fig, use_container_width=True)
    long = dl.allocation_frontier_by_return(pts, wdf)
    if not long.empty:
        st.plotly_chart(px.area(long, x="CAGR", y="Weight", color="Strategy", title="Strategy allocations along the frontier"), use_container_width=True)
    st.subheader("Frontier table")
    display = pts.sort_values(["Efficient", "CAGR"], ascending=[False, False]).head(100)
    display_readable_table(display, use_container_width=True, hide_index=True)
    st.download_button("Download random portfolio cloud CSV", pts.to_csv(index=False).encode("utf-8"), "efficient_frontier_cloud.csv", "text/csv")
    if not long.empty:
        st.download_button("Download frontier allocations CSV", long.to_csv(index=False).encode("utf-8"), "frontier_allocations.csv", "text/csv")
    st.session_state["last_frontier_points"] = pts
    st.session_state["last_frontier_weights"] = wdf


def render_robustness_lab(base_returns: pd.DataFrame) -> None:
    st.header("Robustness Lab")
    st.caption("This tests whether optimized weights are stable across years. Unstable optimizers often pick yesterday's winner and fail out of sample.")
    c1, c2 = st.columns(2)
    max_w = c1.slider("Max weight per strategy", 0.10, 1.00, 0.50, 0.05, key="robust_maxw")
    min_obs = c2.slider("Minimum observations per year", 20, 252, 63, 10)
    metrics, weights = dl.robustness_by_period(base_returns, max_weight=float(max_w), min_obs=int(min_obs))
    if metrics.empty:
        st.warning("Not enough period data for robustness analysis.")
        return
    st.subheader("Year-by-year optimizer performance")
    display_readable_table(metrics, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(metrics, x="Period", y="Sharpe", color="Portfolio", barmode="group", title="Sharpe by year and optimizer"), use_container_width=True)
    c2.plotly_chart(px.bar(metrics, x="Period", y="Max Drawdown", color="Portfolio", barmode="group", title="Max drawdown by year and optimizer"), use_container_width=True)
    if not weights.empty:
        st.subheader("Allocation stability")
        port = st.selectbox("Optimizer for allocation stability", sorted(weights["Portfolio"].unique()))
        sub = weights[weights["Portfolio"] == port]
        st.plotly_chart(px.bar(sub, x="Period", y="Weight", color="Strategy", title=f"Allocation stability: {port}"), use_container_width=True)
        stability = sub.groupby("Strategy")["Weight"].agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
        display_readable_table(stability, use_container_width=True)
        st.download_button("Download robustness weights CSV", weights.to_csv(index=False).encode("utf-8"), "robustness_weights.csv", "text/csv")
    st.download_button("Download robustness metrics CSV", metrics.to_csv(index=False).encode("utf-8"), "robustness_metrics.csv", "text/csv")
    st.session_state["last_robustness_metrics"] = metrics
    st.session_state["last_robustness_weights"] = weights


def render_portfolio_decision_page(base_returns: pd.DataFrame, weights: pd.Series, mc_returns: Optional[pd.DataFrame] = None) -> None:
    st.header("Which Portfolio Should I Run?")
    st.caption("Transparent decision engine combining historical metrics, correlation safety, stress robustness, and Monte Carlo downside. It is not a black box: all components are shown.")
    mc_returns = _safe_universe(mc_returns, base_returns)
    candidates = _candidate_dict_from_state(base_returns)
    c1, c2, c3, c4 = st.columns(4)
    n_sims = c1.slider("MC sims per candidate", 100, 2500, 400, 100, key="decision_mc_sims")
    horizon = c2.slider("MC horizon years", 1, 20, 5, 1, key="decision_mc_horizon")
    block = c3.slider("MC block length", 1, 52, 8, 1, key="decision_mc_block")
    dd_limit = c4.slider("DD breach threshold", -0.80, -0.05, -0.30, 0.05, key="decision_dd")
    metrics_rows = []
    for name, w in candidates.items():
        m = pl.portfolio_metrics(base_returns, w, name)
        m["Concentration HHI"] = float((w ** 2).sum())
        metrics_rows.append(m)
    metrics = pd.DataFrame(metrics_rows).set_index("Portfolio") if metrics_rows else pd.DataFrame()
    run = st.button("Run candidate Monte Carlo comparison", type="primary", use_container_width=True)
    if run or "last_candidate_mc" in st.session_state:
        if run:
            with st.spinner("Running Monte Carlo across candidate portfolios…"):
                mc = dl.candidate_monte_carlo_comparison(mc_returns, candidates, n_sims=int(n_sims), horizon_years=int(horizon), block_len=int(block), dd_limit=float(dd_limit))
            st.session_state["last_candidate_mc"] = mc
        mc = st.session_state.get("last_candidate_mc", pd.DataFrame())
    else:
        mc = pd.DataFrame()
    corr_req = st.session_state.get("last_corr_requirements")
    if corr_req is None or not isinstance(corr_req, pd.DataFrame) or corr_req.empty:
        corr_req = dl.marginal_correlation_requirements(base_returns, weights)
    breakpoints = st.session_state.get("last_corr_breakpoints")
    if breakpoints is None or not isinstance(breakpoints, pd.DataFrame) or breakpoints.empty:
        breakpoints = dl.correlation_breakpoint_curve(base_returns, weights)
    decision = dl.final_decision_table(metrics, corr_req, breakpoints, mc if not mc.empty else None)
    st.session_state["last_decision_table"] = decision
    st.subheader("Final decision table")
    display_readable_table(decision, use_container_width=True)
    if not decision.empty:
        top = decision.index[0]
        st.success(f"Current best candidate under the selected assumptions: **{top}**. This is a recommendation under model assumptions, not a guarantee.")
        fig = px.bar(decision.reset_index(), x="Portfolio", y=["Return Score", "Risk Score", "Risk-Adjusted Score", "MC Downside Score", "Correlation Safety Backdrop"], title="Decision score components", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(px.bar(decision.reset_index(), x="Portfolio", y="Final Score", color="Decision", title="Final score ranking"), use_container_width=True)
    if not mc.empty:
        st.subheader("Candidate Monte Carlo comparison")
        display_readable_table(mc, use_container_width=True)
        mcf = mc.reset_index()
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.scatter(mcf, x="5% Max DD", y="Median CAGR", size="Median Final Wealth", hover_name="Portfolio", title="MC downside vs median CAGR"), use_container_width=True)
        c2.plotly_chart(px.bar(mcf.sort_values("5% CAGR", ascending=False), x="Portfolio", y="5% CAGR", title="Monte Carlo 5% CAGR by candidate"), use_container_width=True)
    sheets = {"Decision": decision, "Historical Metrics": metrics, "MC Comparison": mc, "Correlation Requirements": corr_req, "Breakpoints": breakpoints}
    try:
        xbytes = dl.excel_workbook_bytes(sheets)
        st.download_button("Download portfolio decision workbook XLSX", xbytes, file_name="portfolio_decision_workbook.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    except Exception as exc:
        st.warning(f"Excel export unavailable: {exc}")

def render_benchmark_factor_page(returns_df: pd.DataFrame, native_benchmarks: pd.DataFrame) -> None:
    st.header("Benchmark / Factor Analysis")
    factor_upload = st.file_uploader("Optional benchmark/factor CSV (Date + price/return columns)", type=["csv"], key="factor_csv_v3")
    factor_df = native_benchmarks.copy()
    if factor_upload is not None:
        extra, err = ap.parse_factor_csv(factor_upload)
        if err:
            st.error(f"Could not parse factor CSV: {err}")
        elif not extra.empty:
            factor_df = pd.concat([factor_df, extra], axis=1)
    if factor_df.empty:
        st.warning("No benchmark equity columns or uploaded factor CSVs are available. Upload a factor CSV to enable this page.")
        return
    factor_df = factor_df.loc[:, ~factor_df.T.duplicated()]
    strat = st.selectbox("Strategy / portfolio", returns_df.columns, index=list(returns_df.columns).index("Composite Portfolio") if "Composite Portfolio" in returns_df.columns else 0)
    bench = st.selectbox("Benchmark/factor", factor_df.columns)
    aligned = pd.concat([returns_df[strat], factor_df[bench]], axis=1).dropna()
    if len(aligned) < 10:
        st.warning("Not enough overlapping strategy/benchmark observations.")
        return
    beta = ap.beta_to_benchmark(aligned.iloc[:, 0], aligned.iloc[:, 1])
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    c1, c2, c3 = st.columns(3)
    c1.metric("Static beta", f"{beta:.3f}" if pd.notna(beta) else "n/a")
    c2.metric("Correlation", f"{corr:.3f}" if pd.notna(corr) else "n/a")
    c3.metric("Overlap", f"{len(aligned)} observations")
    window = st.slider("Rolling beta/correlation window", 21, 504, 63, key="market_window_v3")
    roll_beta = ap.beta_to_benchmark(aligned.iloc[:, 0], aligned.iloc[:, 1], window=window)
    roll_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll_beta.index, y=roll_beta, name="Rolling beta"))
    fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, name="Rolling correlation"))
    fig.update_layout(title=f"Rolling market relationship: {strat} vs {bench}")
    st.plotly_chart(fig, use_container_width=True)
    factors = st.multiselect("Factor benchmark columns", factor_df.columns, default=[bench])
    if factors:
        coeffs, stats_dict = ap.factor_regression(returns_df[strat], factor_df[factors])
        if not coeffs.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Regression R²", f"{stats_dict.get('R2', np.nan):.2%}")
            m2.metric("Residual vol", f"{stats_dict.get('Residual Vol', np.nan):.2%}")
            m3.metric("Annual alpha", f"{stats_dict.get('Annual Alpha', np.nan):.2%}")
            st.dataframe(coeffs.style.format("{:.6f}"), use_container_width=True)
            coef_df = coeffs.reset_index().rename(columns={"index": "Factor"})
            st.plotly_chart(px.bar(coef_df, x="Factor", y="Coefficient", title="Regression coefficients"), use_container_width=True)
            if "T-Stat" in coef_df.columns:
                st.plotly_chart(px.bar(coef_df, x="Factor", y="T-Stat", title="Regression t-statistics"), use_container_width=True)
        else:
            st.warning("Regression could not be computed with the selected factor set.")


def render_exposure_page(allocations: Dict[str, pd.DataFrame], composite_weights: pd.Series) -> None:
    st.header("Exposure Analysis")
    if not allocations:
        st.warning("No allocation CSVs were available.")
        return
    names = list(allocations.keys())
    strat = st.selectbox("Strategy allocation", names)
    alloc = allocations.get(strat, pd.DataFrame())
    if alloc.empty:
        st.warning("No holdings for this strategy.")
        return
    stats_dict = ap.concentration_stats(alloc)
    c1, c2, c3 = st.columns(3)
    c1.metric("Top 1 weight", f"{stats_dict['Top 1']:.2%}")
    c2.metric("Top 3 weight", f"{stats_dict['Top 3']:.2%}")
    c3.metric("HHI concentration", f"{stats_dict['HHI']:.3f}")
    left, right = st.columns([1, 1])
    left.dataframe(alloc.style.format({"Weight": "{:.2%}"}), use_container_width=True, hide_index=True)
    right.plotly_chart(px.pie(alloc[alloc["Weight"] > 0], values="Weight", names="Ticker", title=f"Latest holdings: {strat}", hole=0.35), use_container_width=True)
    st.plotly_chart(px.bar(alloc.sort_values("Weight", ascending=False).head(40), x="Ticker", y="Weight", title=f"Top holdings: {strat}"), use_container_width=True)

    st.subheader("Composite latest exposure")
    # Map selected strategy weights to allocation names by direct name, falling back equal.
    adj = {}
    for name in names:
        adj[name] = float(composite_weights.get(name, 0.0)) if name in composite_weights.index else 0.0
    if sum(adj.values()) <= 0:
        adj = {name: 1 / len(names) for name in names}
    combined: Dict[str, float] = {}
    for name, w in adj.items():
        df = allocations.get(name, pd.DataFrame())
        if df.empty:
            continue
        for _, row in df.iterrows():
            combined[row["Ticker"]] = combined.get(row["Ticker"], 0.0) + float(row["Weight"]) * w
    combined_df = pd.DataFrame([{"Ticker": k, "Weight": v} for k, v in combined.items()]).sort_values("Weight", ascending=False)
    if not combined_df.empty:
        cc1, cc2 = st.columns([1, 1])
        cc1.dataframe(combined_df.style.format({"Weight": "{:.2%}"}), use_container_width=True, hide_index=True)
        cc2.plotly_chart(px.pie(combined_df.head(25), names="Ticker", values="Weight", title="Composite top exposure", hole=0.35), use_container_width=True)
        st.plotly_chart(px.bar(combined_df.head(40), x="Ticker", y="Weight", title="Composite top holdings"), use_container_width=True)

    st.subheader("Holdings overlap matrix")
    tickers = sorted({t for df in allocations.values() if not df.empty for t in df["Ticker"].tolist()})
    if tickers and len(names) > 1:
        mat = pd.DataFrame(0.0, index=names, columns=tickers)
        for name, df in allocations.items():
            if df.empty:
                continue
            for _, row in df.iterrows():
                mat.loc[name, row["Ticker"]] = float(row["Weight"])
        overlap = pd.DataFrame(index=names, columns=names, dtype=float)
        for a in names:
            for b in names:
                overlap.loc[a, b] = float(np.minimum(mat.loc[a], mat.loc[b]).sum())
        st.plotly_chart(px.imshow(overlap, text_auto=".2f", color_continuous_scale="Blues", title="Holdings overlap by weight"), use_container_width=True)

    st.subheader("Optional holdings risk contribution")
    risk_csv = st.file_uploader("Optional asset returns/prices CSV", type=["csv"], key="risk_asset_csv_v3")
    if risk_csv is not None:
        asset_rets, err = ap.parse_factor_csv(risk_csv)
        if err:
            st.error(err)
        else:
            rc = ap.holdings_risk_contribution(alloc.set_index("Ticker")["Weight"], asset_rets)
            if not rc.empty:
                rc_df = rc.rename("Risk Contribution").reset_index().rename(columns={"index": "Ticker"})
                st.dataframe(rc_df.style.format({"Risk Contribution": "{:.2%}"}), use_container_width=True, hide_index=True)
                st.plotly_chart(px.bar(rc_df, x="Ticker", y="Risk Contribution", title="Holdings risk contribution"), use_container_width=True)
            else:
                st.warning("No overlap between allocation tickers and uploaded asset columns.")


def _format_html_table(df: pd.DataFrame) -> str:
    """HTML table with readable financial formats for exported reports."""
    if df is None or df.empty:
        return "<p><em>No data available.</em></p>"
    d = df.copy()
    for col in d.columns:
        if hasattr(d[col], "dtype") and d[col].dtype.kind in "fcbi":
            low = str(col).lower()
            if any(h in low for h in PERCENT_HINTS) and not any(h in low for h in ("observations", "count")):
                d[col] = d[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            elif any(h in low for h in RATIO_HINTS):
                d[col] = d[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            else:
                d[col] = d[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    return d.to_html(index=True, escape=False, classes="report-table")


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def render_research_report(returns_df: pd.DataFrame, allocations: Dict[str, pd.DataFrame], base_returns: pd.DataFrame, weights: pd.Series, universe_labels: Optional[Dict[str, str]] = None) -> None:
    st.header("Reports & Downloads")
    selected_default = list(returns_df.columns).index("Composite Portfolio") if "Composite Portfolio" in returns_df.columns else 0
    selected = st.selectbox("Main strategy/portfolio for report detail", returns_df.columns, index=selected_default)
    universe_labels = universe_labels or st.session_state.get("data_universe_modes", {})

    wdf = pd.DataFrame({"Strategy": weights.index, "Weight": weights.values}).sort_values("Weight", ascending=False)
    summary = ap.perf_summary(returns_df)
    selected_returns = returns_df[selected].dropna() if selected in returns_df.columns else pd.Series(dtype=float)
    selected_summary = ap.perf_summary(pd.DataFrame({selected: selected_returns})) if not selected_returns.empty else pd.DataFrame()
    corr_req = st.session_state.get("last_corr_requirements", pd.DataFrame())
    pair_corr = st.session_state.get("last_pairwise_corr", pd.DataFrame())
    corr_safety = st.session_state.get("last_corr_safety", pd.DataFrame())
    candidate_metrics = st.session_state.get("portfolio_candidate_metrics", pd.DataFrame())
    decision_table = st.session_state.get("last_decision_table", pd.DataFrame())
    mc_table = st.session_state.get("last_candidate_mc", pd.DataFrame())

    # Downloadable CSVs first, so the page is useful even if HTML rendering is slow.
    c0, c1, c2 = st.columns(3)
    c0.download_button("Download active portfolio weights CSV", wdf.to_csv(index=False).encode("utf-8"), file_name="active_portfolio_weights.csv", mime="text/csv", use_container_width=True)
    active = pl.portfolio_returns(base_returns, weights, "Composite Portfolio")
    c1.download_button("Download active composite returns CSV", active.to_csv().encode("utf-8"), file_name="composite_portfolio_returns.csv", mime="text/csv", use_container_width=True)
    if not candidate_metrics.empty:
        c2.download_button("Download candidate metrics CSV", candidate_metrics.to_csv().encode("utf-8"), file_name="portfolio_candidate_metrics.csv", mime="text/csv", use_container_width=True)

    css = """
    <style>
      body{font-family:Inter,Arial,Helvetica,sans-serif;margin:28px;color:#18212b;background:#fff;}
      h1{color:#0b2545;margin-bottom:4px} h2{color:#123a63;border-bottom:2px solid #e6edf5;padding-bottom:6px;margin-top:34px;}
      .subtitle{color:#697386;margin-bottom:20px}.kpi{display:inline-block;vertical-align:top;background:#f8fafc;border:1px solid #d9e2ec;border-radius:10px;padding:12px 16px;margin:6px;min-width:150px}
      .kpi b{color:#123a63}.report-table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 22px 0}.report-table th{background:#eef4fa;color:#123a63;font-weight:700}.report-table th,.report-table td{border:1px solid #d6dee7;padding:7px;text-align:right}.report-table td:first-child,.report-table th:first-child{text-align:left}.warn{background:#fff7e6;border:1px solid #ffd591;border-radius:8px;padding:12px;margin:12px 0}.note{background:#eef7ff;border:1px solid #b6ddff;border-radius:8px;padding:12px;margin:12px 0}
    </style>
    """
    parts = ["<html><head><meta charset='utf-8'><title>Portfolio Research Memo</title><script src='https://cdn.plot.ly/plotly-2.27.0.min.js'></script>", css, "</head><body>"]
    parts.append("<h1>Portfolio Research Memo</h1>")
    parts.append(f"<div class='subtitle'>Main report object: <b>{selected}</b>. Generated from backtest outputs and active portfolio weights.</div>")
    if universe_labels:
        mode_df = pd.DataFrame([{"Role": k.title(), "Universe": v} for k, v in universe_labels.items()])
        parts.append("<h2>Evidence Universes</h2><p class='note'>These settings control which periods were used for reporting, optimization, validation, simulation, and stress. Separating them helps reduce self-deception from optimizing and judging on the same period.</p>")
        parts.append(_format_html_table(mode_df.set_index("Role")))

    if not selected_summary.empty:
        srow = selected_summary.iloc[0]
        for k in ["CAGR", "Volatility", "Max Drawdown", "Sharpe", "Sortino", "Calmar", "cVaR 95"]:
            if k in srow.index and pd.notna(srow[k]):
                val = f"{srow[k]:.2%}" if k not in {"Sharpe", "Sortino", "Calmar"} else f"{srow[k]:.2f}"
                parts.append(f"<div class='kpi'><b>{k}</b><br>{val}</div>")

    parts.append("<h2>Selected Portfolio Weights</h2>")
    parts.append(_format_html_table(wdf.set_index("Strategy")))
    parts.append(_fig_html(px.pie(wdf, names="Strategy", values="Weight", title="Selected portfolio weights", hole=0.35)))
    parts.append(_fig_html(px.bar(wdf, x="Strategy", y="Weight", title="Selected portfolio weights")))

    if not returns_df.empty:
        parts.append("<h2>Equity Curves and Drawdowns</h2>")
        parts.append(_fig_html(px.line((1 + returns_df.fillna(0)).cumprod(), title="Equity curves", log_y=True)))
        parts.append(_fig_html(px.line(ap.drawdowns(returns_df), title="Drawdowns")))
    if not summary.empty:
        parts.append("<h2>All Strategy / Portfolio Metrics</h2>")
        parts.append(_format_html_table(summary))
        sdf = summary.reset_index()
        parts.append(_fig_html(px.scatter(sdf, x="Max Drawdown", y="CAGR", size="Volatility", hover_name="Strategy", title="Risk / return map")))
        parts.append(_fig_html(px.bar(sdf.sort_values("Sharpe", ascending=False), x="Strategy", y="Sharpe", title="Sharpe ranking")))

    if selected in returns_df.columns:
        tdd = ap.top_drawdowns(returns_df[selected], top_n=15)
        parts.append(f"<h2>Drawdown Anatomy: {selected}</h2>")
        if not tdd.empty:
            parts.append(_format_html_table(tdd.set_index("Start")))
            parts.append(_fig_html(px.bar(tdd, x="Start", y="Depth", title="Top drawdowns")))
        monthly = ap.monthly_returns_table(returns_df[selected])
        yearly = ap.yearly_returns(returns_df[selected])
        if not monthly.empty:
            parts.append("<h2>Monthly Return Heatmap</h2>")
            parts.append(_fig_html(px.imshow(monthly, text_auto=".1%", color_continuous_scale="RdYlGn", aspect="auto", title="Monthly returns")))
        if not yearly.empty:
            parts.append("<h2>Yearly Returns</h2>")
            parts.append(_fig_html(px.bar(yearly, title="Yearly returns")))

    if returns_df.shape[1] > 1:
        corr = returns_df.corr()
        parts.append("<h2>Correlation Structure</h2>")
        parts.append(_fig_html(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation matrix")))
        if isinstance(corr_safety, pd.DataFrame) and not corr_safety.empty:
            parts.append("<h2>Correlation Safety Margins</h2>")
            parts.append("<p class='note'>Positive values imply observed correlations are below the approximate useful-correlation threshold. Negative values warn that a pair may be too similar for the chosen objective.</p>")
            parts.append(_fig_html(px.imshow(corr_safety, text_auto=".2f", color_continuous_scale="RdYlGn", title="Correlation safety margin")))
        if isinstance(corr_req, pd.DataFrame) and not corr_req.empty:
            parts.append("<h2>Strategy-to-Portfolio Correlation Requirements</h2>")
            parts.append(_format_html_table(corr_req.set_index("Strategy") if "Strategy" in corr_req.columns else corr_req))

    if isinstance(candidate_metrics, pd.DataFrame) and not candidate_metrics.empty:
        parts.append("<h2>Candidate Portfolio Comparison</h2>")
        parts.append(_format_html_table(candidate_metrics))
        cm = candidate_metrics.reset_index()
        if "Decision Score" in cm.columns:
            parts.append(_fig_html(px.bar(cm.sort_values("Decision Score", ascending=False), x="Portfolio", y="Decision Score", title="Candidate decision score")))
    if isinstance(decision_table, pd.DataFrame) and not decision_table.empty:
        parts.append("<h2>Final Portfolio Decision</h2>")
        parts.append(_format_html_table(decision_table))
        dd = decision_table.reset_index()
        if "Final Score" in dd.columns:
            parts.append(_fig_html(px.bar(dd.sort_values("Final Score", ascending=False), x="Portfolio", y="Final Score", color="Decision", title="Final portfolio ranking")))
    if isinstance(mc_table, pd.DataFrame) and not mc_table.empty:
        parts.append("<h2>Monte Carlo Candidate Comparison</h2>")
        parts.append(_format_html_table(mc_table))
        mt = mc_table.reset_index()
        if "5% CAGR" in mt.columns:
            parts.append(_fig_html(px.bar(mt.sort_values("5% CAGR", ascending=False), x="Portfolio", y="5% CAGR", title="Monte Carlo 5% CAGR by candidate")))

    alloc_table = ap.latest_allocation_table(allocations)
    if not alloc_table.empty:
        parts.append("<h2>Latest Strategy Allocations</h2>")
        parts.append(_format_html_table(alloc_table.set_index("Strategy") if "Strategy" in alloc_table.columns else alloc_table))
        top = alloc_table.groupby("Ticker")["Weight"].sum().sort_values(ascending=False).head(25).reset_index()
        parts.append(_fig_html(px.bar(top, x="Ticker", y="Weight", title="Combined top holdings")))
    parts.append("</body></html>")
    html = "\n".join(parts)

    st.download_button("Download full research memo HTML", html.encode("utf-8"), file_name="portfolio_research_memo.html", mime="text/html", use_container_width=True)
    sheets = {
        "Metrics": summary,
        "Weights": wdf,
        "Candidate Metrics": candidate_metrics,
        "Decision": decision_table,
        "Monte Carlo": mc_table,
        "Correlation Requirements": corr_req,
        "Pairwise Corr": pair_corr,
        "Corr Safety": corr_safety,
        "Frontier": st.session_state.get("last_frontier_points", pd.DataFrame()),
        "Frontier Weights": st.session_state.get("last_frontier_weights", pd.DataFrame()),
        "Robustness Metrics": st.session_state.get("last_robustness_metrics", pd.DataFrame()),
        "Robustness Weights": st.session_state.get("last_robustness_weights", pd.DataFrame()),
    }
    try:
        st.download_button("Download formatted research workbook XLSX", dl.excel_workbook_bytes(sheets), file_name="portfolio_research_workbook.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    except Exception as exc:
        st.warning(f"Workbook export failed: {exc}")
    components.html(html, height=1100, scrolling=True)


def render_analytics_suite(run: BacktestRun) -> None:
    raw_returns, native_benchmarks, _eq_map = ap.load_returns_from_outputs(run.output_files, run.index)
    allocations = ap.load_latest_allocations(run.output_files, run.index)
    if raw_returns.empty:
        st.info("No strategy equity CSVs were available for analytics.")
        return

    st.header("Research Workbench")
    align_mode = st.radio("Return alignment", ["Common overlap only", "Native inception dates", "Full history with cash for missing"], horizontal=True)
    aligned_returns = ap.align_returns(raw_returns, align_mode).dropna(how="all")
    if aligned_returns.empty:
        st.warning("The selected alignment mode left no usable return data.")
        return
    universes = render_data_universe_panel(aligned_returns)
    analysis_returns = _safe_universe(universes.get("analysis"), aligned_returns)
    optimization_returns = _safe_universe(universes.get("optimization"), analysis_returns)
    validation_returns = _safe_universe(universes.get("validation"), analysis_returns)
    simulation_returns = _safe_universe(universes.get("simulation"), analysis_returns)
    stress_returns = _safe_universe(universes.get("stress"), analysis_returns)
    universe_labels = universes.get("labels", {})
    base_returns = analysis_returns
    returns_df, active_weights = _add_active_composite(base_returns)

    nav = st.radio(
        "Workbench section",
        ["Performance", "Portfolio Lab", "Correlation Requirements", "Efficient Frontier", "Robustness Lab", "Correlation & Stress Lab", "Monte Carlo Lab", "Portfolio Decision", "Benchmark / Factors", "Exposure", "Drawdowns & Calendars", "Reports & Downloads"],
        horizontal=True,
    )

    with st.expander("Active portfolio weights", expanded=False):
        awdf = pd.DataFrame({"Strategy": active_weights.index, "Weight": active_weights.values}).sort_values("Weight", ascending=False)
        caw1, caw2 = st.columns([1, 1])
        caw1.dataframe(readable_styler(awdf), use_container_width=True, hide_index=True)
        caw2.plotly_chart(px.pie(awdf, names="Strategy", values="Weight", title="Active portfolio weights"), use_container_width=True)

    if nav == "Performance":
        summary = ap.perf_summary(returns_df)
        if not summary.empty:
            display_readable_table(summary, use_container_width=True)
            render_visual_metric_companions(summary)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line((1 + returns_df.fillna(0)).cumprod(), title="Equity curves", log_y=True), use_container_width=True)
        fig_dd = px.line(ap.drawdowns(returns_df), title="Drawdowns")
        fig_dd.update_layout(yaxis_tickformat=".0%")
        c2.plotly_chart(fig_dd, use_container_width=True)
        if returns_df.shape[1] > 1:
            div = ap.diversification_metrics(base_returns)
            cols = st.columns(len(div))
            for col, (k, v) in zip(cols, div.items()):
                col.metric(k, f"{v:.3f}" if pd.notna(v) else "n/a")
        st.subheader("Rolling metrics")
        window = st.slider("Rolling window", 21, 504, 63, key="roll_metrics_window_v3")
        metric_dict = ap.rolling_metrics(returns_df, window=window)
        metric_name = st.selectbox("Metric", list(metric_dict.keys()))
        st.plotly_chart(px.line(metric_dict[metric_name], title=f"{metric_name} ({window} trading days)"), use_container_width=True)

    elif nav == "Portfolio Lab":
        render_portfolio_lab(base_returns, optimization_returns, validation_returns, universe_labels)

    elif nav == "Correlation Requirements":
        render_correlation_requirement_lab(stress_returns, active_weights)

    elif nav == "Efficient Frontier":
        render_efficient_frontier_lab(optimization_returns, active_weights)

    elif nav == "Robustness Lab":
        render_robustness_lab(analysis_returns)

    elif nav == "Correlation & Stress Lab":
        stress_returns_df, _stress_w = _add_active_composite(stress_returns)
        render_correlation_stress_lab(stress_returns, stress_returns_df, active_weights)

    elif nav == "Monte Carlo Lab":
        render_monte_carlo_lab(simulation_returns, active_weights)

    elif nav == "Portfolio Decision":
        render_portfolio_decision_page(validation_returns, active_weights, simulation_returns)

    elif nav == "Benchmark / Factors":
        render_benchmark_factor_page(returns_df, native_benchmarks)

    elif nav == "Exposure":
        render_exposure_page(allocations, active_weights)

    elif nav == "Drawdowns & Calendars":
        strat = st.selectbox("Strategy / portfolio", returns_df.columns, index=list(returns_df.columns).index("Composite Portfolio") if "Composite Portfolio" in returns_df.columns else 0)
        tdd = ap.top_drawdowns(returns_df[strat], top_n=20)
        if not tdd.empty:
            show = tdd.copy()
            show["Depth"] = show["Depth"].map(lambda x: f"{x:.2%}")
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(tdd, x="Start", y="Depth", title=f"Top drawdowns: {strat}"), use_container_width=True)
        c1, c2 = st.columns(2)
        monthly = ap.monthly_returns_table(returns_df[strat])
        if not monthly.empty:
            c1.plotly_chart(px.imshow(monthly, text_auto=".1%", color_continuous_scale="RdYlGn", aspect="auto", title="Monthly return heatmap"), use_container_width=True)
        yearly = ap.yearly_returns(returns_df[strat])
        if not yearly.empty:
            c2.plotly_chart(px.bar(yearly, title="Yearly returns"), use_container_width=True)
        st.plotly_chart(px.line(ap.drawdowns(returns_df[[strat]]), title="Underwater plot"), use_container_width=True)

    elif nav == "Reports & Downloads":
        render_research_report(returns_df, allocations, base_returns, active_weights, universe_labels)


def main() -> None:
    st.title("📈 Quantmage Backtest & Analytics Lab")
    st.caption("Run one or many Quantmage/Rainyboy backtests, compare them, build composites, diagnose correlation/exposure, and download reports.")

    if not ENGINE_PATH.exists() or not CONFIG_DIR.exists():
        st.error("App package is incomplete: qm_native_engine.py or config/ is missing.")
        st.stop()
    if not REPORT_TEMPLATE.exists():
        st.warning("Rain-style report template is missing. The engine may fall back to simple HTML reports.")

    with st.sidebar:
        st.header("Backtest settings")
        benchmark = st.text_input("Benchmark override", value="", help="Optional. Example: SPY, QQQ, TQQQ.")
        chart_benchmark = st.text_input("Extra chart benchmark", value="QQQ", help="Shown only in cumulative return chart.")
        strict = st.checkbox("Strict mode", value=False, help="Fail if the compiler emits warnings.")
        allow_non_letf_proxy = st.checkbox("Allow non-LetfMap heuristic proxies", value=False)
        st.divider()
        st.subheader("Parallel execution")
        parallel_strategy_level = st.checkbox("Parallelize individual strategies", value=True)
        max_workers = st.number_input("Max parallel workers", min_value=1, max_value=8, value=4, step=1)
        retry_failed_sequential = st.checkbox("If parallel run fails, allow manual retry failed only", value=True)
        st.caption("Lower workers if strategies need API/Yahoo extension to avoid rate limits.")
        st.divider()
        use_testfolio_api = st.checkbox("Use Testfolio API extension", value=True)
        use_yahoo_fallback = st.checkbox("Use Yahoo fallback", value=True)
        timeout_sec = st.number_input("API timeout seconds", min_value=5, max_value=180, value=40, step=5)
        retries = st.number_input("API retries", min_value=0, max_value=5, value=2, step=1)
        with st.expander("Optional authentication", expanded=False):
            cookie = st.text_input("Testfolio Cookie", value="", type="password")
            token = st.text_input("Testfolio Bearer Token", value="", type="password")
        st.divider()
        try:
            price_count = len(list((CONFIG_DIR / "prices").glob("*.dat")))
            ticker_count = len(list((CONFIG_DIR / "tickers").glob("*.dat")))
            ext_count = len(list((CONFIG_DIR / "ext_letf").glob("*.dat")))
            st.caption(f"Local caches: {price_count} prices, {ticker_count} tickers, {ext_count} extended LETF files")
        except Exception:
            pass

    st.subheader("Input")
    input_mode = st.radio("Input mode", ["Run Quantmage JSON backtests", "Analyze existing output ZIP"], horizontal=True)

    if input_mode == "Analyze existing output ZIP":
        zip_file = st.file_uploader("Upload prior output ZIP", type=["zip"])
        if zip_file is not None and st.button("Load output ZIP", type="primary", use_container_width=True):
            run = analyze_existing_zip(zip_file)
            st.session_state["last_run"] = run
        run = st.session_state.get("last_run")
        if run is None:
            st.info("Upload an output ZIP generated by this app to analyze it without rerunning backtests.")
            st.stop()
    else:
        uploaded_files = st.file_uploader("Upload Quantmage JSON file(s)", type=["json"], accept_multiple_files=True)
        if not uploaded_files:
            st.info("Upload one or more Quantmage JSON exports or single strategy JSON files to begin.")
            with st.expander("What this app produces"):
                st.markdown("""
                Each run creates downloadable outputs:
                - interactive `.report.html` for each strategy
                - `.equity.csv` and `.allocation.csv` for each strategy
                - diagnostics and warnings
                - multi-strategy analytics addendum
                - performance/correlation/allocation CSV exports
                - one combined ZIP containing every output file
                """)
            st.stop()

        uploaded_payloads: List[Tuple[str, bytes, Any]] = []
        previews: List[pd.DataFrame] = []
        for uploaded in uploaded_files:
            raw = uploaded.getvalue()
            payload, error = parse_uploaded_json(raw)
            if error:
                st.error(f"{uploaded.name}: {error}")
                st.stop()
            uploaded_payloads.append((uploaded.name, raw, payload))
            previews.append(extract_strategy_preview(payload, source_name=uploaded.name))

        preview_df = pd.concat([df for df in previews if not df.empty], ignore_index=True) if previews else pd.DataFrame()
        selected_keys: List[str] = []
        if preview_df.empty:
            st.warning("No Quantmage strategy objects were detected. The engine may fail unless the files are supported formats.")
        else:
            st.subheader("Detected strategies")
            top_only = st.checkbox("Show only top-level/minimum-depth detections", value=False)
            shown = preview_df.copy()
            if top_only and not shown.empty:
                shown = shown[shown["depth"] == shown["depth"].min()]
            labels = [f"{row['source']} | {row['name']} | path={row['path']}" for _, row in shown.iterrows()]
            key_by_label = {label: key for label, key in zip(labels, shown["run_key"].tolist())}
            selected_labels = st.multiselect("Strategies to run", labels, default=labels)
            selected_keys = [key_by_label[x] for x in selected_labels]
            display_cols = ["#", "source", "name", "path", "depth", "benchmark", "trading_type", "rebalance"]
            st.dataframe(shown[display_cols], use_container_width=True, hide_index=True)
            st.caption(f"Selected {len(selected_keys)} of {len(preview_df)} detected strategy object(s).")

        if parallel_strategy_level and len(selected_keys) > 1:
            st.success(f"Parallel strategy mode is on. The app will create up to {int(max_workers)} concurrent worker(s).")
        elif len(uploaded_files) > 1:
            st.info("Multiple uploads will be processed as a batch. Turn on strategy-level parallelism for concurrent execution.")

        run_button = st.button("Run backtest batch", type="primary", use_container_width=True)
        if run_button:
            with st.status("Running backtest engine…", expanded=True) as status:
                progress = st.progress(0.0, text="Starting backtest tasks…")
                try:
                    run = run_backtest_batch(uploaded_payloads, benchmark, chart_benchmark, strict, use_testfolio_api, use_yahoo_fallback, allow_non_letf_proxy, int(timeout_sec), int(retries), cookie, token, parallel_strategy_level, int(max_workers), selected_strategy_keys=selected_keys if selected_keys else None, progress_placeholder=progress)
                    st.session_state["last_run"] = run
                    status.update(label="Backtest batch complete" if run.returncode == 0 else "Backtest batch finished with warnings/errors", state="complete" if run.returncode == 0 else "error")
                except Exception as exc:
                    status.update(label="Backtest failed", state="error")
                    st.exception(exc)

        run: Optional[BacktestRun] = st.session_state.get("last_run")
        if run is None:
            st.stop()

    st.divider()
    st.header("Backtest results")
    if run.returncode != 0:
        st.error("At least one task or wrapper step had warnings/errors. Partial outputs may still be available.")
    else:
        st.success(f"Engine completed successfully. Tasks: {run.task_count}; parallel strategy mode: {run.parallel_strategy_level}.")

    results_nav = st.radio("Results section", ["Diagnostics", "Generated Outputs", "Analytics", "Report Preview", "Logs"], horizontal=True)
    if results_nav == "Diagnostics":
        render_task_status(run)
        render_index(run)
    elif results_nav == "Generated Outputs":
        render_downloads(run)
    elif results_nav == "Analytics":
        render_analytics_suite(run)
    elif results_nav == "Report Preview":
        render_report_preview(run)
    elif results_nav == "Logs":
        st.text_area("stdout", run.stdout, height=420)
        if run.stderr.strip():
            st.text_area("stderr", run.stderr, height=220)


if __name__ == "__main__":
    main()
