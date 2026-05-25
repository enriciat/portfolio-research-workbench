from __future__ import annotations

"""Optional helpers for Rain Java CLI/report parity workflows.

These helpers are intentionally lightweight.  The Streamlit app and Python
engine do not depend on Java at runtime, but local users can call the Rain CLI
as an upstream oracle when a compatible Java runtime is available.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Optional


@dataclass
class RainCliResult:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    command: list[str]


def default_cli_jar(app_root: Path) -> Path:
    return app_root / "config" / "rain" / "BacktestCLI.jar"


def default_report_jar(app_root: Path) -> Path:
    return app_root / "config" / "rain" / "BacktestReport.jar"


def java_available() -> bool:
    return shutil.which("java") is not None


def run_rain_cli(
    input_json: Path,
    output_dir: Path,
    *,
    cli_jar: Optional[Path] = None,
    mode: str = "SYNTHETIC",
    benchmark: str = "",
    timeout_sec: int = 300,
) -> RainCliResult:
    """Run Rain's BacktestCLI.jar if available.

    The exact Java CLI surface can change upstream, so failures are returned as
    structured results instead of raising.  Use this for local parity checks,
    not as the default Streamlit Cloud execution path.
    """
    app_root = Path(__file__).resolve().parents[1]
    jar = Path(cli_jar) if cli_jar else default_cli_jar(app_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-jar", str(jar), "--mode", mode, "-i", str(input_json), "-o", str(output_dir)]
    if benchmark:
        cmd += ["--benchmark", benchmark]
    if not java_available():
        return RainCliResult(False, 127, "", "java executable not found", cmd)
    if not jar.exists():
        return RainCliResult(False, 2, "", f"Rain CLI jar not found: {jar}", cmd)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
        return RainCliResult(proc.returncode == 0, proc.returncode, proc.stdout or "", proc.stderr or "", cmd)
    except subprocess.TimeoutExpired as exc:
        return RainCliResult(False, 124, exc.stdout or "", exc.stderr or f"Timed out after {timeout_sec}s", cmd)
