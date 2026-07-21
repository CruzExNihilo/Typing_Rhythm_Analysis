from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def log_run(project_root: Path, mode: str, session_files: list[Path], output_file: str | None, success: bool) -> None:
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "analysis_run_log.jsonl"
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "session_files": [path.name for path in session_files],
        "output_file": output_file,
        "success": success,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_command(script_path: Path, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(script_path)]
    if args.mode == "single":
        if args.session_file:
            command.extend(args.session_file[:1])
    elif args.mode == "population":
        if args.session_file:
            command.extend(args.session_file)
    elif args.mode == "batch":
        if args.sessions_dir:
            command.extend(["--sessions-dir", args.sessions_dir])
    if args.output_name:
        command.extend(["--output-name", args.output_name])
    if args.dpi:
        command.extend(["--dpi", str(args.dpi)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run typing analysis workflows from the command line.")
    parser.add_argument("mode", choices=["single", "population", "batch"], help="Choose the analysis mode.")
    parser.add_argument("session_file", nargs="*", help="Optional JSON session files or directories to analyze.")
    parser.add_argument("--output-name", help="Custom output filename without extension.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for the generated PDF figure.")
    parser.add_argument("--sessions-dir", default="sessions", help="Directory containing session JSON files for batch mode.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tools_dir = project_root / "analysis_tools"

    if args.mode == "single":
        script = tools_dir / "single_session_analysis.py"
        session_files = [Path(path) for path in args.session_file[:1]] if args.session_file else []
    elif args.mode == "population":
        script = tools_dir / "population_analysis.py"
        session_files = [Path(path) for path in args.session_file] if args.session_file else []
    else:
        script = tools_dir / "batch_analysis.py"
        session_files = []

    if not script.exists():
        raise FileNotFoundError(f"Required script not found: {script}")

    command = build_command(script, args)
    print(f"Running analysis: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=project_root, check=True)
        output_name = args.output_name or {
            "single": f"typing_speed_analysis_{datetime.now().strftime('%Y%m%d')}",
            "population": f"typing_speed_population_analysis_{datetime.now().strftime('%Y%m%d')}",
            "batch": f"batch_analysis_{datetime.now().strftime('%Y%m%d')}",
        }[args.mode]
        log_run(project_root, args.mode, session_files, f"{output_name}.pdf", True)
    except subprocess.CalledProcessError as exc:
        log_run(project_root, args.mode, session_files, None, False)
        raise exc


if __name__ == "__main__":
    main()
